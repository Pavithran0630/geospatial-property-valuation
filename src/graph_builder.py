"""
Graph Builder Module
=====================

Week 3 – Stage 2: Construct a K-Nearest Neighbours (KNN) spatial graph
from the graph-ready housing dataset produced by Stage 1.

Every property becomes a **node**.  Two nodes are connected by a directed edge
when one property is among the K nearest spatial neighbours of the other
(distance measured in kilometres using the Haversine formula on lat/long).

Outputs
-------
data/graph_edges.csv   – Edge list: source node, target node, Haversine distance.
data/adjacency.pkl     – Adjacency list: dict mapping node_id → list of neighbour node_ids.

No embeddings are generated here.  Embedding generation is delegated to later stages.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAT_COL = "lat"
LON_COL = "long"
NODE_ID_COL = "node_id"

# Earth's mean radius in kilometres (used to convert radians → km)
EARTH_RADIUS_KM = 6371.0

# Default file paths (relative to project root, matching existing conventions)
DEFAULT_INPUT_PATH  = "data/graph_ready_data.csv"
DEFAULT_EDGES_PATH  = "data/graph_edges.csv"
DEFAULT_ADJ_PATH    = "data/adjacency.pkl"

# Default number of nearest neighbours per node
DEFAULT_K = 5


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_graph_ready_data(path: str = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """
    Load the graph-ready dataset produced by the Stage 1 preprocessing pipeline.

    Parameters
    ----------
    path : str
        Path to the graph-ready CSV file.
        Defaults to ``data/graph_ready_data.csv``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing at minimum: node_id, lat, long.

    Raises
    ------
    FileNotFoundError
        If the dataset file is not found at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph-ready dataset not found at '{path}'. "
            "Please run src/graph_preprocessing.py first."
        )

    print(f"[INFO] Loading graph-ready dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} nodes  |  {df.shape[1]} columns")
    return df


def save_edge_list(edges_df: pd.DataFrame, path: str = DEFAULT_EDGES_PATH) -> None:
    """
    Save the edge list DataFrame to a CSV file.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        DataFrame with columns: source, target, distance_km.
    path : str
        Destination CSV path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    edges_df.to_csv(path, index=False)
    print(f"[INFO] Edge list saved to: {path}  ({len(edges_df):,} edges)")


def save_adjacency_list(adjacency: dict, path: str = DEFAULT_ADJ_PATH) -> None:
    """
    Persist the adjacency list dictionary to disk using pickle.

    Parameters
    ----------
    adjacency : dict
        Mapping of node_id (int) → list of neighbour node_ids (list[int]).
    path : str
        Destination pickle path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(adjacency, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Adjacency list saved to: {path}  ({len(adjacency):,} nodes)")


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------

def extract_coordinates(df: pd.DataFrame) -> np.ndarray:
    """
    Extract latitude and longitude as a NumPy array in radians.

    The BallTree with the Haversine metric requires coordinates in radians.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing lat/long columns.

    Returns
    -------
    numpy.ndarray of shape (n_nodes, 2)
        Coordinates in radians: [[lat_rad, lon_rad], ...].
    """
    coords_deg = df[[LAT_COL, LON_COL]].values.astype(np.float64)
    coords_rad = np.radians(coords_deg)
    print(f"[INFO] Extracted {len(coords_rad):,} coordinate pairs (converted to radians).")
    return coords_rad


# ---------------------------------------------------------------------------
# KNN graph construction
# ---------------------------------------------------------------------------

def build_knn_index(coords_rad: np.ndarray) -> BallTree:
    """
    Build a BallTree spatial index on the radian coordinates for efficient
    Haversine-distance nearest-neighbour queries.

    Parameters
    ----------
    coords_rad : numpy.ndarray of shape (n_nodes, 2)
        Coordinates in radians.

    Returns
    -------
    sklearn.neighbors.BallTree
        Fitted BallTree using the Haversine metric.
    """
    print("[INFO] Building BallTree spatial index (Haversine metric)...")
    tree = BallTree(coords_rad, metric="haversine")
    print("[INFO] BallTree index built successfully.")
    return tree


def query_knn(
    tree: BallTree,
    coords_rad: np.ndarray,
    k: int = DEFAULT_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Query the BallTree to find the K nearest spatial neighbours for every node.

    The query asks for k+1 neighbours because the nearest neighbour of any
    point is itself (distance = 0); that self-loop is excluded from the results.

    Parameters
    ----------
    tree : BallTree
        Fitted spatial index.
    coords_rad : numpy.ndarray of shape (n_nodes, 2)
        Coordinates in radians.
    k : int
        Number of nearest neighbours to find per node.

    Returns
    -------
    distances_km : numpy.ndarray of shape (n_nodes, k)
        Haversine distances in kilometres to each neighbour.
    indices : numpy.ndarray of shape (n_nodes, k)
        Row indices (= node_ids) of the k nearest neighbours.
    """
    print(f"[INFO] Querying {k}-nearest neighbours for {len(coords_rad):,} nodes...")

    # k+1 because the node itself is always the closest point (dist=0)
    distances_rad, indices = tree.query(coords_rad, k=k + 1)

    # Convert radians to kilometres and strip the self-reference (column 0)
    distances_km = distances_rad[:, 1:] * EARTH_RADIUS_KM
    indices = indices[:, 1:]

    print(
        f"[INFO] KNN query complete. "
        f"Mean neighbour distance: {distances_km.mean():.2f} km  |  "
        f"Max: {distances_km.max():.2f} km"
    )
    return distances_km, indices


# ---------------------------------------------------------------------------
# Edge list and adjacency list construction
# ---------------------------------------------------------------------------

def build_edge_list(
    node_ids: np.ndarray,
    distances_km: np.ndarray,
    neighbour_indices: np.ndarray,
) -> pd.DataFrame:
    """
    Convert the raw KNN query results into a tidy edge-list DataFrame.

    Each row represents a directed edge: source_node → target_node.
    The edge carries the Haversine distance between the two property locations.

    Parameters
    ----------
    node_ids : numpy.ndarray of shape (n_nodes,)
        The node_id value for each row in the dataset.
    distances_km : numpy.ndarray of shape (n_nodes, k)
        Haversine distances (km) to each of the k neighbours.
    neighbour_indices : numpy.ndarray of shape (n_nodes, k)
        Row indices of the k nearest neighbours.

    Returns
    -------
    pandas.DataFrame
        Columns: source (int), target (int), distance_km (float).
    """
    print("[INFO] Building edge list...")

    n_nodes, k = distances_km.shape

    # Repeat each source node_id k times (once per neighbour)
    sources  = np.repeat(node_ids, k)

    # Map neighbour row-indices back to their actual node_ids
    targets  = node_ids[neighbour_indices.flatten()]

    # Flatten the distance matrix to match sources/targets
    distances = distances_km.flatten()

    edges_df = pd.DataFrame({
        "source":      sources.astype(int),
        "target":      targets.astype(int),
        "distance_km": np.round(distances, 6),
    })

    print(f"[INFO] Edge list built: {len(edges_df):,} directed edges  "
          f"({n_nodes:,} nodes × {k} neighbours).")
    return edges_df


def build_adjacency_list(edges_df: pd.DataFrame) -> dict:
    """
    Build an adjacency list from the edge-list DataFrame.

    The adjacency list maps every node_id to the list of its direct neighbours.
    This is the standard representation used by graph algorithms and GNN frameworks.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge list with columns: source, target, distance_km.

    Returns
    -------
    dict
        ``{node_id (int): [neighbour_id, ...]}``
    """
    print("[INFO] Building adjacency list from edge list...")

    adjacency = (
        edges_df
        .groupby("source")["target"]
        .apply(list)
        .to_dict()
    )

    print(f"[INFO] Adjacency list built: {len(adjacency):,} entries.")
    return adjacency


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_graph(
    input_path:  str = DEFAULT_INPUT_PATH,
    edges_path:  str = DEFAULT_EDGES_PATH,
    adj_path:    str = DEFAULT_ADJ_PATH,
    k:           int = DEFAULT_K,
) -> tuple[pd.DataFrame, dict]:
    """
    End-to-end pipeline that constructs the KNN spatial graph.

    Steps
    -----
    1. Load the graph-ready dataset.
    2. Extract lat/long coordinates and convert to radians.
    3. Build a BallTree spatial index (Haversine metric).
    4. Query k-nearest neighbours for every node.
    5. Build a directed edge list (source, target, distance_km).
    6. Build an adjacency list (node_id → list of neighbours).
    7. Save edge list to CSV and adjacency list to pickle.

    Parameters
    ----------
    input_path : str
        Path to the graph-ready dataset CSV.
    edges_path : str
        Output path for the edge-list CSV.
    adj_path : str
        Output path for the adjacency-list pickle.
    k : int
        Number of nearest spatial neighbours per node.

    Returns
    -------
    edges_df : pandas.DataFrame
        Full edge list.
    adjacency : dict
        Adjacency list mapping node_id → neighbour list.
    """
    print("=" * 60)
    print("Graph Builder Pipeline")
    print(f"K-Nearest Neighbours  : k = {k}")
    print("=" * 60)

    # Step 1 – Load the graph-ready dataset
    df = load_graph_ready_data(input_path)

    # Step 2 – Extract coordinate matrix (in radians for Haversine)
    coords_rad = extract_coordinates(df)

    # Preserve the node_id array for mapping row-indices → node IDs
    node_ids = df[NODE_ID_COL].values

    # Step 3 – Build BallTree spatial index
    tree = build_knn_index(coords_rad)

    # Step 4 – Find k nearest neighbours for every node
    distances_km, neighbour_indices = query_knn(tree, coords_rad, k=k)

    # Step 5 – Build the directed edge list
    edges_df = build_edge_list(node_ids, distances_km, neighbour_indices)

    # Step 6 – Build the adjacency list
    adjacency = build_adjacency_list(edges_df)

    # Step 7 – Persist both outputs to disk
    save_edge_list(edges_df, edges_path)
    save_adjacency_list(adjacency, adj_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\nGraph Construction Summary")
    print("-" * 60)
    print(f"Total nodes           : {len(df):,}")
    print(f"Neighbours per node   : {k}")
    print(f"Total directed edges  : {len(edges_df):,}")
    print(f"Avg distance (km)     : {edges_df['distance_km'].mean():.2f}")
    print(f"Min distance (km)     : {edges_df['distance_km'].min():.4f}")
    print(f"Max distance (km)     : {edges_df['distance_km'].max():.2f}")
    print(f"\nEdge list saved to    : {edges_path}")
    print(f"Adjacency list saved to: {adj_path}")
    print("\nGraph construction completed successfully.")
    print("=" * 60)

    return edges_df, adjacency


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_graph(
        input_path = DEFAULT_INPUT_PATH,
        edges_path = DEFAULT_EDGES_PATH,
        adj_path   = DEFAULT_ADJ_PATH,
        k          = DEFAULT_K,
    )
