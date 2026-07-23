"""
Spatial Embeddings Module
==========================

Week 3 – Stage 4: Generate spatial node embeddings representing neighbourhood
context for every property in the KNN graph.

Approach: Node2Vec
------------------
Node2Vec performs biased random walks on the graph and feeds the resulting
walk sequences into a Word2Vec (skip-gram) model.  The resulting vectors
capture both local neighbourhood structure (BFS-like, p < 1) and structural
equivalence (DFS-like, q < 1).

Fallback: Neighbourhood Feature Aggregation
-------------------------------------------
If node2vec is unavailable (import error / version conflict), a lightweight
hand-crafted embedding is used instead:
  - Each node's embedding is the row-wise aggregation (mean) of the
    graph-ready feature matrix for that node and all its direct neighbours.
  - This still encodes genuine spatial / neighbourhood context.

Outputs
-------
data/spatial_embeddings.csv  – (n_nodes, 1 + embedding_dim) — node_id + embedding dims.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EDGES_PATH     = "data/graph_edges.csv"
DEFAULT_ADJ_PATH       = "data/adjacency.pkl"
DEFAULT_NODES_PATH     = "data/graph_ready_data.csv"
DEFAULT_EMBEDDINGS_OUT = "data/spatial_embeddings.csv"

# Node2Vec / random-walk hyper-parameters
DEFAULT_EMBEDDING_DIM   = 64    # number of embedding dimensions (configurable)
DEFAULT_WALK_LENGTH     = 30    # steps per random walk
DEFAULT_NUM_WALKS       = 10    # walks per node
DEFAULT_P               = 1.0   # return parameter (1 = no bias)
DEFAULT_Q               = 1.0   # in-out parameter (1 = no bias)
DEFAULT_WORKERS         = 4     # parallel workers for Word2Vec
DEFAULT_EPOCHS          = 5     # Word2Vec training epochs
DEFAULT_WINDOW          = 5     # Word2Vec context window size
RANDOM_SEED             = 42


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_edge_list(path: str = DEFAULT_EDGES_PATH) -> pd.DataFrame:
    """
    Load the edge-list CSV produced by Stage 2.

    Parameters
    ----------
    path : str
        Path to graph_edges.csv.

    Returns
    -------
    pandas.DataFrame
        Columns: source (int), target (int), distance_km (float).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Edge list not found at '{path}'. "
            "Please run src/graph_builder.py first."
        )
    print(f"[INFO] Loading edge list from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} edges.")
    return df


def load_adjacency_list(path: str = DEFAULT_ADJ_PATH) -> dict:
    """
    Load the adjacency-list pickle produced by Stage 2.

    Parameters
    ----------
    path : str
        Path to adjacency.pkl.

    Returns
    -------
    dict
        ``{node_id (int): [neighbour_id, ...]}``
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Adjacency list not found at '{path}'. "
            "Please run src/graph_builder.py first."
        )
    print(f"[INFO] Loading adjacency list from: {path}")
    with open(path, "rb") as f:
        adj = pickle.load(f)
    print(f"[INFO] Adjacency list loaded: {len(adj):,} nodes.")
    return adj


def load_node_data(path: str = DEFAULT_NODES_PATH) -> pd.DataFrame:
    """
    Load the graph-ready node dataset (features + coordinates).

    Parameters
    ----------
    path : str
        Path to graph_ready_data.csv.

    Returns
    -------
    pandas.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Node dataset not found at '{path}'. "
            "Please run src/graph_preprocessing.py first."
        )
    print(f"[INFO] Loading node data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} nodes  |  {df.shape[1]} columns.")
    return df


def save_embeddings(embeddings_df: pd.DataFrame,
                    path: str = DEFAULT_EMBEDDINGS_OUT) -> None:
    """
    Save the node embedding matrix to a CSV file.

    The first column is always ``node_id``; subsequent columns are the
    embedding dimensions named ``emb_0``, ``emb_1``, … ``emb_{dim-1}``.

    Parameters
    ----------
    embeddings_df : pandas.DataFrame
        Embedding matrix with node_id as the first column.
    path : str
        Destination CSV path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    embeddings_df.to_csv(path, index=False)
    print(f"[INFO] Embeddings saved to: {path}  "
          f"({len(embeddings_df):,} nodes × {embeddings_df.shape[1] - 1} dims)")


# ---------------------------------------------------------------------------
# Graph construction (NetworkX)
# ---------------------------------------------------------------------------

def build_nx_graph(edges_df: pd.DataFrame):
    """
    Build an undirected NetworkX graph from the edge-list.

    Node2Vec works on an undirected graph; the directed KNN graph is converted
    to undirected so that random walks can traverse edges in both directions.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Columns: source, target, distance_km.

    Returns
    -------
    networkx.Graph
        Undirected, weighted graph.
    """
    import networkx as nx

    print("[INFO] Building undirected NetworkX graph for Node2Vec...")
    G = nx.from_pandas_edgelist(
        edges_df,
        source="source",
        target="target",
        edge_attr="distance_km",
        create_using=nx.Graph(),      # undirected
    )
    print(f"[INFO] Graph ready — nodes: {G.number_of_nodes():,}  |  "
          f"edges: {G.number_of_edges():,}")
    return G


# ---------------------------------------------------------------------------
# Primary approach: Node2Vec
# ---------------------------------------------------------------------------

def generate_node2vec_embeddings(
    G,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    walk_length:   int = DEFAULT_WALK_LENGTH,
    num_walks:     int = DEFAULT_NUM_WALKS,
    p:             float = DEFAULT_P,
    q:             float = DEFAULT_Q,
    workers:       int = DEFAULT_WORKERS,
    epochs:        int = DEFAULT_EPOCHS,
    window:        int = DEFAULT_WINDOW,
    seed:          int = RANDOM_SEED,
) -> dict:
    """
    Generate node embeddings using the Node2Vec algorithm.

    Node2Vec performs biased random walks on the graph and trains a Word2Vec
    skip-gram model on the resulting walk sequences.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph to embed.
    embedding_dim : int
        Dimensionality of the output embedding vectors.
    walk_length : int
        Number of steps in each random walk.
    num_walks : int
        Number of random walks starting from each node.
    p : float
        Return parameter – controls likelihood of revisiting a node.
        Higher p = less backtracking.
    q : float
        In-out parameter – controls BFS vs DFS exploration.
        q < 1 → DFS-like (structural); q > 1 → BFS-like (local neighbourhood).
    workers : int
        Parallel threads for Word2Vec training.
    epochs : int
        Training epochs for the Word2Vec model.
    window : int
        Context window size for Word2Vec skip-gram.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{node_id (int): numpy.ndarray of shape (embedding_dim,)}``
    """
    from node2vec import Node2Vec

    print(f"[INFO] Running Node2Vec  "
          f"(dim={embedding_dim}, walks={num_walks}, length={walk_length}, "
          f"p={p}, q={q}) ...")

    # Build the Node2Vec random-walk model
    node2vec_model = Node2Vec(
        G,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        quiet=True,         # suppress walk progress bars in production
    )

    # Train the Word2Vec (skip-gram) model on the generated walks
    print("[INFO] Training Word2Vec on random walks...")
    wv_model = node2vec_model.fit(
        window=window,
        min_count=1,
        batch_words=4,
        epochs=epochs,
        seed=seed,
    )

    # Extract embedding vectors keyed by node_id
    embeddings = {
        int(node): wv_model.wv[str(node)]
        for node in G.nodes()
    }

    print(f"[INFO] Node2Vec embeddings generated for {len(embeddings):,} nodes.")
    return embeddings


# ---------------------------------------------------------------------------
# Fallback approach: Neighbourhood Feature Aggregation
# ---------------------------------------------------------------------------

def generate_neighbourhood_embeddings(
    nodes_df: pd.DataFrame,
    adjacency: dict,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Generate graph-based embeddings via neighbourhood feature aggregation.

    For each node the embedding is constructed as:
        e(v) = concat(
            mean_pool(features of v ∪ neighbours of v),   # neighbourhood context
            PCA-reduced(v's own features)                  # node self-features
        )

    This captures genuine spatial neighbourhood context without an external
    package.

    Parameters
    ----------
    nodes_df : pandas.DataFrame
        Graph-ready node data with node_id as a column.
    adjacency : dict
        ``{node_id: [neighbour_ids]}``
    embedding_dim : int
        Target embedding dimensionality (achieved via PCA on the pooled features).
    seed : int
        Random seed for PCA reproducibility.

    Returns
    -------
    dict
        ``{node_id (int): numpy.ndarray of shape (embedding_dim,)}``
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("[INFO] Falling back to Neighbourhood Feature Aggregation embedding...")

    # ---- Select numeric feature columns (exclude id, date strings) ------
    non_feature_cols = {"node_id", "id", "date"}
    feature_cols = [
        c for c in nodes_df.columns
        if c not in non_feature_cols
        and pd.api.types.is_numeric_dtype(nodes_df[c])
    ]
    print(f"[INFO] Using {len(feature_cols)} numeric features for aggregation.")

    # Build a fast node_id → feature vector lookup (numpy matrix)
    node_ids   = nodes_df["node_id"].values.astype(int)
    feature_mx = nodes_df[feature_cols].values.astype(np.float64)

    # Fill any remaining NaNs with column means before scaling
    col_means = np.nanmean(feature_mx, axis=0)
    nan_mask  = np.isnan(feature_mx)
    feature_mx[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Standardise so all features contribute equally to the mean-pool
    scaler     = StandardScaler()
    feature_mx = scaler.fit_transform(feature_mx)

    # Build row-index lookup: node_id → matrix row
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # ---- Neighbourhood mean-pooling -------------------------------------
    print("[INFO] Aggregating neighbourhood features (mean-pooling)...")
    pooled_mx = np.zeros_like(feature_mx)

    for i, nid in enumerate(node_ids):
        neighbours   = adjacency.get(nid, [])
        nbr_indices  = [id_to_idx[n] for n in neighbours if n in id_to_idx]

        if nbr_indices:
            # Pool node's own features + all neighbour features
            pool_rows       = [i] + nbr_indices
            pooled_mx[i]    = feature_mx[pool_rows].mean(axis=0)
        else:
            # Isolated node: use own features only
            pooled_mx[i]    = feature_mx[i]

    # ---- PCA to target dimensionality -----------------------------------
    actual_dim = min(embedding_dim, pooled_mx.shape[1])
    if actual_dim < embedding_dim:
        print(f"[WARN] Requested dim={embedding_dim} exceeds number of features "
              f"({pooled_mx.shape[1]}). Using dim={actual_dim}.")

    print(f"[INFO] Applying PCA: {pooled_mx.shape[1]} features → {actual_dim} dims...")
    pca        = PCA(n_components=actual_dim, random_state=seed)
    embedding_mx = pca.fit_transform(pooled_mx)

    explained  = pca.explained_variance_ratio_.sum() * 100
    print(f"[INFO] PCA explains {explained:.1f}% of variance in {actual_dim} components.")

    embeddings = {
        int(nid): embedding_mx[i]
        for i, nid in enumerate(node_ids)
    }

    print(f"[INFO] Neighbourhood embeddings generated for {len(embeddings):,} nodes.")
    return embeddings


# ---------------------------------------------------------------------------
# Embedding matrix → DataFrame
# ---------------------------------------------------------------------------

def embeddings_to_dataframe(embeddings: dict) -> pd.DataFrame:
    """
    Convert the node-embedding dictionary to a tidy DataFrame.

    Columns
    -------
    node_id : int
        Unique property identifier.
    emb_0 … emb_{dim-1} : float
        Embedding dimensions.

    Parameters
    ----------
    embeddings : dict
        ``{node_id: ndarray}``

    Returns
    -------
    pandas.DataFrame
        Shape: (n_nodes, 1 + embedding_dim).
    """
    node_ids  = sorted(embeddings.keys())
    dim       = len(next(iter(embeddings.values())))
    col_names = [f"emb_{i}" for i in range(dim)]

    matrix = np.vstack([embeddings[nid] for nid in node_ids])
    df     = pd.DataFrame(matrix, columns=col_names)
    df.insert(0, "node_id", node_ids)

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def generate_spatial_embeddings(
    edges_path:     str   = DEFAULT_EDGES_PATH,
    adj_path:       str   = DEFAULT_ADJ_PATH,
    nodes_path:     str   = DEFAULT_NODES_PATH,
    output_path:    str   = DEFAULT_EMBEDDINGS_OUT,
    embedding_dim:  int   = DEFAULT_EMBEDDING_DIM,
    walk_length:    int   = DEFAULT_WALK_LENGTH,
    num_walks:      int   = DEFAULT_NUM_WALKS,
    p:              float = DEFAULT_P,
    q:              float = DEFAULT_Q,
    workers:        int   = DEFAULT_WORKERS,
    epochs:         int   = DEFAULT_EPOCHS,
    window:         int   = DEFAULT_WINDOW,
    seed:           int   = RANDOM_SEED,
) -> pd.DataFrame:
    """
    End-to-end pipeline for generating spatial node embeddings.

    Tries Node2Vec first; if it cannot be imported, falls back to the
    Neighbourhood Feature Aggregation approach automatically.

    Steps
    -----
    1. Load edge list, adjacency list, and node data.
    2. Build undirected NetworkX graph.
    3. Attempt Node2Vec embedding generation.
       If unavailable, fall back to Neighbourhood Feature Aggregation.
    4. Convert embeddings to a tidy DataFrame.
    5. Save to data/spatial_embeddings.csv.

    Parameters
    ----------
    edges_path : str
        Path to graph_edges.csv.
    adj_path : str
        Path to adjacency.pkl.
    nodes_path : str
        Path to graph_ready_data.csv.
    output_path : str
        Output path for spatial_embeddings.csv.
    embedding_dim : int
        Embedding vector dimensionality (configurable).
    walk_length : int
        Node2Vec walk length.
    num_walks : int
        Node2Vec walks per node.
    p : float
        Node2Vec return parameter.
    q : float
        Node2Vec in-out parameter.
    workers : int
        Word2Vec worker threads.
    epochs : int
        Word2Vec training epochs.
    window : int
        Word2Vec context window.
    seed : int
        Global random seed.

    Returns
    -------
    pandas.DataFrame
        Embedding matrix: (n_nodes, 1 + embedding_dim).
    """
    print("=" * 60)
    print("Spatial Embeddings Pipeline")
    print(f"Embedding dimension : {embedding_dim}")
    print("=" * 60)

    # Step 1 – Load data
    edges_df  = load_edge_list(edges_path)
    adjacency = load_adjacency_list(adj_path)
    nodes_df  = load_node_data(nodes_path)

    # Step 2 – Build undirected graph (needed by Node2Vec)
    G = build_nx_graph(edges_df)

    # Step 3 – Generate embeddings (Node2Vec preferred, fallback if unavailable)
    method_used = "unknown"
    try:
        import node2vec as _n2v_test  # noqa: F401 – just testing importability
        embeddings  = generate_node2vec_embeddings(
            G,
            embedding_dim = embedding_dim,
            walk_length   = walk_length,
            num_walks     = num_walks,
            p             = p,
            q             = q,
            workers       = workers,
            epochs        = epochs,
            window        = window,
            seed          = seed,
        )
        method_used = "Node2Vec (random walks + Word2Vec)"

    except (ImportError, ModuleNotFoundError):
        print("[WARN] node2vec package not available. "
              "Falling back to Neighbourhood Feature Aggregation.")
        embeddings = generate_neighbourhood_embeddings(
            nodes_df,
            adjacency,
            embedding_dim = embedding_dim,
            seed          = seed,
        )
        method_used = "Neighbourhood Feature Aggregation (mean-pool + PCA)"

    # Step 4 – Convert to DataFrame
    print("[INFO] Converting embeddings to DataFrame...")
    embeddings_df = embeddings_to_dataframe(embeddings)

    # Step 5 – Save
    save_embeddings(embeddings_df, output_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\nSpatial Embeddings Summary")
    print("-" * 60)
    print(f"Method used          : {method_used}")
    print(f"Total nodes embedded : {len(embeddings_df):,}")
    print(f"Embedding dimension  : {embeddings_df.shape[1] - 1}")
    print(f"Output shape         : {embeddings_df.shape}")
    print(f"Output saved to      : {output_path}")

    # Quick sanity: check for NaN in embeddings
    nan_count = embeddings_df.iloc[:, 1:].isnull().sum().sum()
    if nan_count > 0:
        print(f"[WARN] {nan_count} NaN values found in embeddings!")
    else:
        print("[INFO] No NaN values detected in embeddings.")

    print("\nSpatial embedding generation completed successfully.")
    print("=" * 60)

    return embeddings_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_spatial_embeddings(
        embedding_dim = DEFAULT_EMBEDDING_DIM,    # change here to reconfigure
    )
