"""
Graph Analysis Module
======================

Week 3 – Stage 3: Analyse the KNN spatial graph built in Stage 2.

Computes the following graph metrics:
- Number of nodes
- Number of edges
- Average degree
- Graph density
- Number of connected components
- Degree distribution

Generates:
- results/graph_statistics.csv    – Key graph metrics as a table.
- results/degree_distribution.png – Histogram of node degree distribution.
- results/graph_visualization.png – Spatial scatter-plot of a sampled subgraph.

No embeddings are generated here.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # headless backend – safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EDGES_PATH = "data/graph_edges.csv"
DEFAULT_ADJ_PATH   = "data/adjacency.pkl"
DEFAULT_NODES_PATH = "data/graph_ready_data.csv"

RESULTS_DIR = "results"

OUT_STATS_CSV  = os.path.join(RESULTS_DIR, "graph_statistics.csv")
OUT_DEGREE_PNG = os.path.join(RESULTS_DIR, "degree_distribution.png")
OUT_VIZ_PNG    = os.path.join(RESULTS_DIR, "graph_visualization.png")

# Number of nodes to sample for the visualization (full graph is too dense)
VIZ_SAMPLE_SIZE = 1000


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

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
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
    Load the graph-ready node dataset (lat/long used for visualization).

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
    print(f"[INFO] Loaded {len(df):,} nodes.")
    return df


# ---------------------------------------------------------------------------
# Graph construction (NetworkX)
# ---------------------------------------------------------------------------

def build_nx_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed NetworkX graph from the edge-list DataFrame.

    Each edge carries a ``weight`` attribute equal to the Haversine distance
    between the two connected properties.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Columns: source, target, distance_km.

    Returns
    -------
    networkx.DiGraph
        Directed graph with weighted edges.
    """
    print("[INFO] Building directed NetworkX graph...")

    G = nx.from_pandas_edgelist(
        edges_df,
        source="source",
        target="target",
        edge_attr="distance_km",
        create_using=nx.DiGraph(),
    )

    print(f"[INFO] Graph built — nodes: {G.number_of_nodes():,}  |  edges: {G.number_of_edges():,}")
    return G


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_basic_metrics(G: nx.DiGraph) -> dict:
    """
    Compute fundamental graph-level statistics.

    Metrics
    -------
    - num_nodes         : Total number of property nodes.
    - num_edges         : Total number of directed edges.
    - avg_in_degree     : Average number of incoming edges per node.
    - avg_out_degree    : Average number of outgoing edges per node.
    - density           : Fraction of possible edges that actually exist.

    Parameters
    ----------
    G : networkx.DiGraph

    Returns
    -------
    dict
        Key → value mapping of computed metrics.
    """
    print("[INFO] Computing basic graph metrics...")

    n = G.number_of_nodes()
    e = G.number_of_edges()

    # Average in-degree and out-degree
    in_degrees  = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    metrics = {
        "num_nodes":       n,
        "num_edges":       e,
        "avg_in_degree":   round(float(np.mean(in_degrees)),  4),
        "avg_out_degree":  round(float(np.mean(out_degrees)), 4),
        "density":         round(nx.density(G), 8),
    }

    print(f"  Nodes          : {metrics['num_nodes']:,}")
    print(f"  Edges          : {metrics['num_edges']:,}")
    print(f"  Avg in-degree  : {metrics['avg_in_degree']}")
    print(f"  Avg out-degree : {metrics['avg_out_degree']}")
    print(f"  Density        : {metrics['density']}")

    return metrics


def compute_connected_components(G: nx.DiGraph) -> dict:
    """
    Compute weakly and strongly connected component statistics.

    Weakly connected components treat edges as undirected.
    Strongly connected components respect edge direction.

    Parameters
    ----------
    G : networkx.DiGraph

    Returns
    -------
    dict
        Component counts and sizes of the largest components.
    """
    print("[INFO] Computing connected components...")

    # Weakly connected (ignore edge direction)
    weak_components  = list(nx.weakly_connected_components(G))
    n_weak           = len(weak_components)
    largest_weak_sz  = max(len(c) for c in weak_components)

    # Strongly connected (respect edge direction)
    strong_components = list(nx.strongly_connected_components(G))
    n_strong          = len(strong_components)
    largest_strong_sz = max(len(c) for c in strong_components)

    metrics = {
        "weakly_connected_components":        n_weak,
        "largest_weak_component_size":        largest_weak_sz,
        "strongly_connected_components":      n_strong,
        "largest_strong_component_size":      largest_strong_sz,
    }

    print(f"  Weakly  connected components : {n_weak:,}  (largest: {largest_weak_sz:,} nodes)")
    print(f"  Strongly connected components: {n_strong:,}  (largest: {largest_strong_sz:,} nodes)")

    return metrics


def compute_degree_distribution(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute the out-degree distribution of all nodes.

    Returns a DataFrame summarising how many nodes have each degree value,
    which is used both for statistics and for plotting.

    Parameters
    ----------
    G : networkx.DiGraph

    Returns
    -------
    pandas.DataFrame
        Columns: degree (int), count (int), fraction (float).
    """
    print("[INFO] Computing degree distribution...")

    out_degrees = dict(G.out_degree())
    degree_series = pd.Series(out_degrees, name="degree")

    dist_df = (
        degree_series
        .value_counts()
        .sort_index()
        .reset_index()
    )
    dist_df.columns = ["degree", "count"]
    dist_df["fraction"] = dist_df["count"] / dist_df["count"].sum()

    print(f"  Unique degree values : {len(dist_df)}")
    print(f"  Min degree           : {dist_df['degree'].min()}")
    print(f"  Max degree           : {dist_df['degree'].max()}")
    print(f"  Median degree        : {degree_series.median():.1f}")

    return dist_df, degree_series


# ---------------------------------------------------------------------------
# Output: statistics CSV
# ---------------------------------------------------------------------------

def save_graph_statistics(
    basic_metrics: dict,
    component_metrics: dict,
    degree_series: pd.Series,
    path: str = OUT_STATS_CSV,
) -> pd.DataFrame:
    """
    Consolidate all computed metrics into a single CSV file.

    Parameters
    ----------
    basic_metrics : dict
        From :func:`compute_basic_metrics`.
    component_metrics : dict
        From :func:`compute_connected_components`.
    degree_series : pandas.Series
        Node out-degree values.
    path : str
        Output CSV path.

    Returns
    -------
    pandas.DataFrame
        The statistics table that was saved.
    """
    all_metrics = {**basic_metrics, **component_metrics}

    # Add degree summary stats
    all_metrics["min_degree"]    = int(degree_series.min())
    all_metrics["max_degree"]    = int(degree_series.max())
    all_metrics["median_degree"] = float(degree_series.median())
    all_metrics["std_degree"]    = round(float(degree_series.std()), 4)

    stats_df = pd.DataFrame(
        list(all_metrics.items()),
        columns=["metric", "value"]
    )

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    stats_df.to_csv(path, index=False)
    print(f"[INFO] Graph statistics saved to: {path}")

    return stats_df


# ---------------------------------------------------------------------------
# Output: degree distribution plot
# ---------------------------------------------------------------------------

def plot_degree_distribution(
    dist_df: pd.DataFrame,
    degree_series: pd.Series,
    path: str = OUT_DEGREE_PNG,
) -> None:
    """
    Plot and save the node out-degree distribution as a styled histogram.

    Parameters
    ----------
    dist_df : pandas.DataFrame
        Output of :func:`compute_degree_distribution`.
    degree_series : pandas.Series
        Raw per-node degree values (for statistical overlays).
    path : str
        Destination PNG path.
    """
    print("[INFO] Plotting degree distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    mean_deg   = degree_series.mean()
    median_deg = degree_series.median()

    # ---- Left: bar chart of degree counts --------------------------------
    ax1 = axes[0]
    ax1.set_facecolor("#1a1d27")

    colors = cm.plasma(np.linspace(0.2, 0.9, len(dist_df)))
    bars = ax1.bar(
        dist_df["degree"],
        dist_df["count"],
        color=colors,
        edgecolor="#2a2d3a",
        linewidth=0.5,
        zorder=3,
    )

    ax1.axvline(mean_deg,   color="#f59e0b", linewidth=1.8, linestyle="--",
                label=f"Mean = {mean_deg:.1f}", zorder=4)
    ax1.axvline(median_deg, color="#10b981", linewidth=1.8, linestyle=":",
                label=f"Median = {median_deg:.1f}", zorder=4)

    ax1.set_xlabel("Out-Degree", color="#cbd5e1", fontsize=11)
    ax1.set_ylabel("Number of Nodes", color="#cbd5e1", fontsize=11)
    ax1.set_title("Degree Distribution (Count)", color="#f1f5f9", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="#94a3b8")
    ax1.spines["bottom"].set_color("#334155")
    ax1.spines["left"].set_color("#334155")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", color="#334155", linewidth=0.5, zorder=0)
    ax1.legend(fontsize=9, facecolor="#1e2433", edgecolor="#334155",
               labelcolor="#cbd5e1")

    # ---- Right: fraction (normalised) plot --------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#1a1d27")

    ax2.bar(
        dist_df["degree"],
        dist_df["fraction"],
        color=colors,
        edgecolor="#2a2d3a",
        linewidth=0.5,
        zorder=3,
    )
    ax2.set_xlabel("Out-Degree", color="#cbd5e1", fontsize=11)
    ax2.set_ylabel("Fraction of Nodes", color="#cbd5e1", fontsize=11)
    ax2.set_title("Degree Distribution (Normalised)", color="#f1f5f9",
                  fontsize=13, fontweight="bold")
    ax2.tick_params(colors="#94a3b8")
    ax2.spines["bottom"].set_color("#334155")
    ax2.spines["left"].set_color("#334155")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", color="#334155", linewidth=0.5, zorder=0)

    fig.suptitle(
        "KNN Spatial Graph — Out-Degree Distribution",
        color="#f8fafc", fontsize=15, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Degree distribution plot saved to: {path}")


# ---------------------------------------------------------------------------
# Output: graph visualization (sampled subgraph)
# ---------------------------------------------------------------------------

def plot_graph_visualization(
    G: nx.DiGraph,
    nodes_df: pd.DataFrame,
    sample_size: int = VIZ_SAMPLE_SIZE,
    path: str = OUT_VIZ_PNG,
) -> None:
    """
    Render and save a spatial scatter-plot of a sampled subgraph.

    Properties are plotted at their true geographic coordinates (lon, lat).
    Edges are drawn as faint lines between connected nodes.  Node colour
    encodes out-degree; node size is uniform for clarity.

    Parameters
    ----------
    G : networkx.DiGraph
        Full graph (subgraph is sampled from this).
    nodes_df : pandas.DataFrame
        Node metadata containing node_id, lat, long.
    sample_size : int
        Number of nodes to visualise.
    path : str
        Destination PNG path.
    """
    print(f"[INFO] Generating graph visualization (sample: {sample_size:,} nodes)...")

    # --- Sample a connected subset for clarity ----------------------------
    np.random.seed(42)
    all_nodes = list(G.nodes())
    sampled_nodes = set(np.random.choice(all_nodes, size=min(sample_size, len(all_nodes)),
                                         replace=False).tolist())
    subG = G.subgraph(sampled_nodes).copy()

    # --- Build position dict: node_id → (longitude, latitude) ------------
    coord_map = nodes_df.set_index("node_id")[["long", "lat"]].to_dict("index")
    pos = {
        n: (coord_map[n]["long"], coord_map[n]["lat"])
        for n in subG.nodes()
        if n in coord_map
    }

    # Filter to nodes with known positions
    valid_nodes = [n for n in subG.nodes() if n in pos]
    subG = subG.subgraph(valid_nodes).copy()

    # --- Degree for colour encoding ---------------------------------------
    degrees      = dict(subG.out_degree())
    node_list    = list(subG.nodes())
    degree_vals  = np.array([degrees[n] for n in node_list], dtype=float)

    # --- Draw ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#080c14")

    # Draw edges first (under nodes)
    nx.draw_networkx_edges(
        subG, pos,
        ax=ax,
        edge_color="#1e3a5f",
        alpha=0.35,
        width=0.5,
        arrows=False,
    )

    # Draw nodes coloured by degree
    sc = nx.draw_networkx_nodes(
        subG, pos,
        ax=ax,
        nodelist=node_list,
        node_color=degree_vals,
        cmap=cm.plasma,
        node_size=18,
        alpha=0.85,
        linewidths=0.3,
        edgecolors="#0ea5e9",
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cm.plasma,
        norm=plt.Normalize(vmin=degree_vals.min(), vmax=degree_vals.max()),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Out-Degree", color="#94a3b8", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    # Labels & styling
    ax.set_xlabel("Longitude", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Latitude",  color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.spines["bottom"].set_color("#1e293b")
    ax.spines["left"].set_color("#1e293b")
    ax.spines["top"].set_color("#1e293b")
    ax.spines["right"].set_color("#1e293b")
    ax.grid(color="#1e293b", linewidth=0.5, alpha=0.6)

    ax.set_title(
        f"KNN Spatial Graph — Sampled Subgraph ({len(subG.nodes()):,} nodes, "
        f"{len(subG.edges()):,} edges)",
        color="#f1f5f9", fontsize=14, fontweight="bold", pad=14,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Graph visualization saved to: {path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def analyse_graph(
    edges_path:     str = DEFAULT_EDGES_PATH,
    adj_path:       str = DEFAULT_ADJ_PATH,
    nodes_path:     str = DEFAULT_NODES_PATH,
    stats_out:      str = OUT_STATS_CSV,
    degree_out:     str = OUT_DEGREE_PNG,
    viz_out:        str = OUT_VIZ_PNG,
    viz_sample:     int = VIZ_SAMPLE_SIZE,
) -> dict:
    """
    End-to-end graph analysis pipeline.

    Steps
    -----
    1. Load edge list, adjacency list, and node data.
    2. Build a directed NetworkX graph.
    3. Compute basic graph metrics (nodes, edges, degree, density).
    4. Compute connected-component statistics.
    5. Compute degree distribution.
    6. Save graph_statistics.csv.
    7. Plot and save degree_distribution.png.
    8. Plot and save graph_visualization.png.

    Parameters
    ----------
    edges_path : str
        Path to graph_edges.csv.
    adj_path : str
        Path to adjacency.pkl.
    nodes_path : str
        Path to graph_ready_data.csv (for coordinates).
    stats_out : str
        Output path for graph_statistics.csv.
    degree_out : str
        Output path for degree_distribution.png.
    viz_out : str
        Output path for graph_visualization.png.
    viz_sample : int
        Number of nodes to include in the visualization subgraph.

    Returns
    -------
    dict
        Combined dictionary of all computed metrics.
    """
    print("=" * 60)
    print("Graph Analysis Pipeline")
    print("=" * 60)

    # Step 1 – Load data
    edges_df  = load_edge_list(edges_path)
    _adj      = load_adjacency_list(adj_path)   # loaded but used implicitly via G
    nodes_df  = load_node_data(nodes_path)

    # Step 2 – Build NetworkX directed graph
    G = build_nx_graph(edges_df)

    # Step 3 – Basic metrics
    basic_metrics = compute_basic_metrics(G)

    # Step 4 – Connected components
    component_metrics = compute_connected_components(G)

    # Step 5 – Degree distribution
    dist_df, degree_series = compute_degree_distribution(G)

    # Step 6 – Save statistics CSV
    stats_df = save_graph_statistics(
        basic_metrics, component_metrics, degree_series, stats_out
    )

    # Step 7 – Degree distribution plot
    plot_degree_distribution(dist_df, degree_series, degree_out)

    # Step 8 – Graph visualization
    plot_graph_visualization(G, nodes_df, viz_sample, viz_out)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    all_metrics = {**basic_metrics, **component_metrics}

    print("\nGraph Analysis Summary")
    print("-" * 60)
    for metric, value in all_metrics.items():
        print(f"  {metric:<42}: {value}")
    print(f"\nOutputs")
    print(f"  {stats_out}")
    print(f"  {degree_out}")
    print(f"  {viz_out}")
    print("\nGraph analysis completed successfully.")
    print("=" * 60)

    return all_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyse_graph()
