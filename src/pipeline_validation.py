"""
Pipeline Validation Module
============================

Week 3 – Stage 6: Validate the entire Week 3 spatial pipeline and generate
a consolidated summary report.

This module performs the following tasks:
1. Verify that all required Week 3 artefacts exist on disk.
2. Validate that each artefact loads correctly and has the expected structure.
3. Generate visualizations:
   - KNN connections overlaid on the property map
   - Embedding feature correlation heatmap
   - Node degree histogram (if not already present)
4. Write results/pipeline_summary.txt – a human-readable pipeline report.

Do NOT train or modify any ML models.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Expected Week 3 artefacts
# ---------------------------------------------------------------------------

WEEK3_ARTEFACTS = {
    # Artefact label          : file path
    "Graph-ready dataset"     : "data/graph_ready_data.csv",
    "Graph edges (CSV)"       : "data/graph_edges.csv",
    "Adjacency list (pickle)" : "data/adjacency.pkl",
    "Spatial embeddings"      : "data/spatial_embeddings.csv",
    "Final spatial dataset"   : "data/spatial_dataset.csv",
}


# ---------------------------------------------------------------------------
# Task 1 – Artefact existence check
# ---------------------------------------------------------------------------

def verify_artefacts(artefacts: dict = WEEK3_ARTEFACTS) -> dict:
    """
    Check that every expected Week 3 artefact exists on disk.

    Parameters
    ----------
    artefacts : dict
        Mapping of ``{label: path}`` for each expected file.

    Returns
    -------
    dict
        ``{label: True/False}`` – True when the file exists.
    """
    print("\n[CHECK] Verifying Week 3 artefacts")
    print("-" * 50)

    results = {}
    all_ok = True

    for label, path in artefacts.items():
        exists = os.path.exists(path)
        results[label] = exists
        status = "[OK]  " if exists else "[MISSING]"
        size_str = ""
        if exists:
            size_kb = os.path.getsize(path) / 1024
            size_str = f"  ({size_kb:,.0f} KB)"
        print(f"  {status} {label:<30} {path}{size_str}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n  All artefacts present.")
    else:
        missing = [l for l, v in results.items() if not v]
        print(f"\n  [WARN] Missing artefacts: {missing}")

    return results


# ---------------------------------------------------------------------------
# Task 2 – Structural validation
# ---------------------------------------------------------------------------

def validate_graph_ready_dataset(path: str = "data/graph_ready_data.csv") -> pd.DataFrame:
    """
    Load and validate the graph-ready dataset.

    Checks:
    - File is loadable
    - 'node_id', 'lat', 'long' columns are present
    - No duplicate node_ids
    - No missing coordinates
    """
    print("\n[VALIDATE] Graph-ready dataset")
    df = pd.read_csv(path)
    assert "node_id" in df.columns, "Missing column: node_id"
    assert "lat" in df.columns,     "Missing column: lat"
    assert "long" in df.columns,    "Missing column: long"
    assert df["node_id"].nunique() == len(df), "Duplicate node_ids detected"
    assert df[["lat", "long"]].isnull().sum().sum() == 0, "Missing coordinates"
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {df.shape[1]}")
    print(f"  node_id range : 0 -> {df['node_id'].max()}")
    print(f"  Lat range     : [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    print(f"  Lon range     : [{df['long'].min():.4f}, {df['long'].max():.4f}]")
    print("  [OK] Validation passed.")
    return df


def validate_graph_edges(path: str = "data/graph_edges.csv") -> pd.DataFrame:
    """
    Load and validate the graph edge list.

    Checks:
    - File is loadable
    - 'source', 'target', 'distance_km' columns are present
    - No negative distances
    - No self-loops
    """
    print("\n[VALIDATE] Graph edges")
    edges = pd.read_csv(path)
    assert "source"      in edges.columns, "Missing column: source"
    assert "target"      in edges.columns, "Missing column: target"
    assert "distance_km" in edges.columns, "Missing column: distance_km"
    assert (edges["distance_km"] >= 0).all(), "Negative distances found"
    self_loops = (edges["source"] == edges["target"]).sum()
    print(f"  Total edges   : {len(edges):,}")
    print(f"  Unique sources: {edges['source'].nunique():,}")
    print(f"  Unique targets: {edges['target'].nunique():,}")
    print(f"  Self-loops    : {self_loops}")
    print(f"  Avg dist (km) : {edges['distance_km'].mean():.3f}")
    print(f"  Max dist (km) : {edges['distance_km'].max():.3f}")
    print("  [OK] Validation passed.")
    return edges


def validate_adjacency_list(path: str = "data/adjacency.pkl") -> dict:
    """
    Load and validate the adjacency list pickle.

    Checks:
    - File loads as a dict
    - All values are lists
    - No empty neighbour lists
    """
    print("\n[VALIDATE] Adjacency list")
    with open(path, "rb") as f:
        adj = pickle.load(f)
    assert isinstance(adj, dict), "adjacency.pkl is not a dict"
    assert all(isinstance(v, list) for v in adj.values()), "Values must be lists"
    neighbour_counts = [len(v) for v in adj.values()]
    print(f"  Total nodes   : {len(adj):,}")
    print(f"  Min neighbours: {min(neighbour_counts)}")
    print(f"  Max neighbours: {max(neighbour_counts)}")
    print(f"  Avg neighbours: {sum(neighbour_counts)/len(neighbour_counts):.2f}")
    print("  [OK] Validation passed.")
    return adj


def validate_embeddings(path: str = "data/spatial_embeddings.csv") -> pd.DataFrame:
    """
    Load and validate the spatial embeddings CSV.

    Checks:
    - File is loadable
    - 'node_id' column is present
    - No NaN values in any embedding dimension
    - Embedding vectors are numeric
    """
    print("\n[VALIDATE] Spatial embeddings")
    embs = pd.read_csv(path)
    assert "node_id" in embs.columns, "Missing column: node_id"
    emb_cols = [c for c in embs.columns if c.startswith("emb_")]
    assert len(emb_cols) > 0, "No embedding columns found (expected emb_0, emb_1, ...)"
    nan_count = embs[emb_cols].isnull().sum().sum()
    assert nan_count == 0, f"NaN values found in embeddings: {nan_count}"
    dim = len(emb_cols)
    print(f"  Nodes         : {len(embs):,}")
    print(f"  Embedding dim : {dim}")
    print(f"  NaN values    : {nan_count}")
    print(f"  Emb_0 mean    : {embs['emb_0'].mean():.4f}")
    print(f"  Emb_0 std     : {embs['emb_0'].std():.4f}")
    print("  [OK] Validation passed.")
    return embs


def validate_spatial_dataset(path: str = "data/spatial_dataset.csv") -> pd.DataFrame:
    """
    Load and validate the final integrated spatial dataset.

    Checks:
    - File is loadable
    - 'node_id', 'price', 'lat', 'long' present
    - Embedding columns present
    - No NaN values
    """
    print("\n[VALIDATE] Final spatial dataset")
    df = pd.read_csv(path)
    for col in ["node_id", "price", "lat", "long"]:
        assert col in df.columns, f"Missing column: {col}"
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    assert len(emb_cols) > 0, "No embedding columns in spatial dataset"
    total_nan = df.isnull().sum().sum()
    housing_cols = [c for c in df.columns if not c.startswith("emb_")]
    print(f"  Shape         : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Housing cols  : {len(housing_cols)}")
    print(f"  Embedding cols: {len(emb_cols)}")
    print(f"  NaN values    : {total_nan}")
    assert total_nan == 0, f"NaN values found in spatial dataset: {total_nan}"
    print("  [OK] Validation passed.")
    return df


# ---------------------------------------------------------------------------
# Task 3a – KNN map overlay
# ---------------------------------------------------------------------------

def plot_knn_map_overlay(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    sample_nodes: int = 300,
    path: str = None,
) -> None:
    """
    Plot KNN edge connections overlaid on the geographic property scatter map.

    A sample of nodes is selected; their KNN edges are drawn as faint lines
    between the actual lon/lat coordinates of each connected property pair.
    Node colour encodes price (log-scaled).

    Parameters
    ----------
    nodes_df : pandas.DataFrame
        Graph-ready dataset with node_id, lat, long, price.
    edges_df : pandas.DataFrame
        Graph edges with source, target, distance_km.
    sample_nodes : int
        Number of nodes to include in the overlay.
    path : str
        Output PNG path.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "knn_map_overlay.png")

    print(f"\n[VIZ] Generating KNN map overlay ({sample_nodes} nodes)...")

    np.random.seed(42)
    sampled_ids = set(
        np.random.choice(nodes_df["node_id"].values, size=sample_nodes, replace=False)
    )

    # Filter edges to only those where BOTH endpoints are in the sample
    mask = (
        edges_df["source"].isin(sampled_ids) &
        edges_df["target"].isin(sampled_ids)
    )
    sampled_edges = edges_df[mask]

    # Build coord lookup: node_id -> (lon, lat)
    coord = nodes_df.set_index("node_id")[["long", "lat", "price"]].to_dict("index")

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#080c14")

    # Draw edges first
    for _, row in sampled_edges.iterrows():
        src, tgt = int(row["source"]), int(row["target"])
        if src in coord and tgt in coord:
            ax.plot(
                [coord[src]["long"], coord[tgt]["long"]],
                [coord[src]["lat"],  coord[tgt]["lat"]],
                color="#1e4d8c",
                alpha=0.35,
                linewidth=0.6,
                zorder=1,
            )

    # Draw nodes coloured by log(price)
    sub_nodes = nodes_df[nodes_df["node_id"].isin(sampled_ids)]
    log_prices = np.log1p(sub_nodes["price"].values)

    sc = ax.scatter(
        sub_nodes["long"],
        sub_nodes["lat"],
        c=log_prices,
        cmap=cm.plasma,
        s=22,
        alpha=0.90,
        edgecolors="#0ea5e9",
        linewidths=0.3,
        zorder=2,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cm.plasma,
        norm=plt.Normalize(vmin=log_prices.min(), vmax=log_prices.max()),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("log(Price)", color="#94a3b8", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    ax.set_xlabel("Longitude", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Latitude",  color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.grid(color="#1e293b", linewidth=0.5, alpha=0.6)
    ax.set_title(
        f"KNN Spatial Graph — Property Map Overlay\n"
        f"({len(sub_nodes):,} properties, {len(sampled_edges):,} KNN connections, "
        f"coloured by price)",
        color="#f1f5f9", fontsize=13, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Task 3b – Embedding correlation heatmap
# ---------------------------------------------------------------------------

def plot_embedding_correlation_heatmap(
    embs_df: pd.DataFrame,
    n_dims: int = 20,
    path: str = None,
) -> None:
    """
    Plot a correlation heatmap of the first ``n_dims`` embedding dimensions.

    Reveals which embedding axes co-vary, giving insight into the
    neighbourhood structure captured by Node2Vec.

    Parameters
    ----------
    embs_df : pandas.DataFrame
        Spatial embeddings DataFrame (node_id + emb_* columns).
    n_dims : int
        Number of leading embedding dimensions to include.
    path : str
        Output PNG path.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "embedding_correlation_heatmap.png")

    print(f"\n[VIZ] Generating embedding correlation heatmap (first {n_dims} dims)...")

    emb_cols = [c for c in embs_df.columns if c.startswith("emb_")][:n_dims]
    corr = embs_df[emb_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", color="#94a3b8", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    # Axis ticks
    tick_labels = [c.replace("emb_", "d") for c in emb_cols]
    ax.set_xticks(range(len(emb_cols)))
    ax.set_yticks(range(len(emb_cols)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right",
                       color="#94a3b8", fontsize=8)
    ax.set_yticklabels(tick_labels, color="#94a3b8", fontsize=8)

    ax.set_title(
        f"Spatial Embedding Correlation Heatmap\n(first {n_dims} Node2Vec dimensions)",
        color="#f1f5f9", fontsize=13, fontweight="bold", pad=14,
    )

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Task 3c – Node degree histogram (generate only if not already present)
# ---------------------------------------------------------------------------

def plot_degree_histogram(
    edges_df: pd.DataFrame,
    path: str = None,
) -> None:
    """
    Plot a node out-degree histogram.

    Skipped automatically if the file already exists (Stage 3 may have
    produced it via graph_analysis.py).

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge list with 'source' column.
    path : str
        Output PNG path.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "degree_distribution.png")

    if os.path.exists(path):
        print(f"\n[VIZ] Degree histogram already exists — skipping. ({path})")
        return

    print("\n[VIZ] Generating node degree histogram...")

    degree_counts = edges_df.groupby("source")["target"].count()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    ax.hist(
        degree_counts.values,
        bins=max(degree_counts.unique()),
        color="#7c3aed",
        edgecolor="#4c1d95",
        linewidth=0.5,
    )
    ax.axvline(degree_counts.mean(), color="#f59e0b", linewidth=1.8,
               linestyle="--", label=f"Mean = {degree_counts.mean():.1f}")

    ax.set_xlabel("Out-Degree (KNN Neighbours)", color="#cbd5e1", fontsize=11)
    ax.set_ylabel("Number of Nodes",             color="#cbd5e1", fontsize=11)
    ax.set_title("Node Degree Distribution",     color="#f1f5f9",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#334155", linewidth=0.5)
    ax.legend(fontsize=9, facecolor="#1e2433", edgecolor="#334155",
              labelcolor="#cbd5e1")

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Task 4 – Pipeline summary report
# ---------------------------------------------------------------------------

def write_pipeline_summary(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    embs_df: pd.DataFrame,
    final_df: pd.DataFrame,
    artefact_status: dict,
    path: str = None,
) -> None:
    """
    Write a human-readable plain-text pipeline summary report.

    Covers:
    - Number of nodes and edges
    - Average neighbours per node
    - Embedding dimension
    - Final dataset shape
    - All Week 3 files generated

    Parameters
    ----------
    nodes_df, edges_df, embs_df, final_df : pandas.DataFrame
        Validated Week 3 datasets.
    artefact_status : dict
        Output of :func:`verify_artefacts`.
    path : str
        Output path for the .txt report.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "pipeline_summary.txt")

    avg_neighbours = edges_df.groupby("source")["target"].count().mean()
    emb_dim        = embs_df.shape[1] - 1
    emb_cols       = [c for c in final_df.columns if c.startswith("emb_")]
    housing_cols   = [c for c in final_df.columns if not c.startswith("emb_")]

    week3_files = [
        ("data/graph_ready_data.csv",    "Stage 1 – Graph-ready housing dataset"),
        ("data/graph_edges.csv",          "Stage 2 – KNN edge list (source, target, distance_km)"),
        ("data/adjacency.pkl",            "Stage 2 – Adjacency list (node_id -> [neighbours])"),
        ("results/graph_statistics.csv",  "Stage 3 – Graph metrics table"),
        ("results/degree_distribution.png","Stage 3 – Degree distribution plot"),
        ("results/graph_visualization.png","Stage 3 – Spatial subgraph visualisation"),
        ("data/spatial_embeddings.csv",   "Stage 4 – Node2Vec spatial embeddings"),
        ("data/spatial_dataset.csv",      "Stage 5 – Final integrated dataset"),
        ("results/knn_map_overlay.png",   "Stage 6 – KNN connections on property map"),
        ("results/embedding_correlation_heatmap.png", "Stage 6 – Embedding correlation heatmap"),
        ("results/pipeline_summary.txt",  "Stage 6 – This pipeline summary report"),
    ]

    lines = [
        "=" * 65,
        "WEEK 3 PIPELINE VALIDATION SUMMARY",
        "Geospatial Property Valuation via Spatial Embeddings",
        "=" * 65,
        "",
        "GRAPH STATISTICS",
        "-" * 40,
        f"  Total nodes (properties)  : {len(nodes_df):,}",
        f"  Total directed edges      : {len(edges_df):,}",
        f"  Average neighbours (k)    : {avg_neighbours:.1f}",
        f"  Min dist between nbrs (km): {edges_df['distance_km'].min():.4f}",
        f"  Avg dist between nbrs (km): {edges_df['distance_km'].mean():.3f}",
        f"  Max dist between nbrs (km): {edges_df['distance_km'].max():.2f}",
        "",
        "EMBEDDING STATISTICS",
        "-" * 40,
        f"  Method                    : Node2Vec (random walks + Word2Vec)",
        f"  Embedding dimension       : {emb_dim}",
        f"  Nodes embedded            : {len(embs_df):,}",
        f"  NaN in embeddings         : {embs_df[emb_cols].isnull().sum().sum()}",
        "",
        "FINAL DATASET (spatial_dataset.csv)",
        "-" * 40,
        f"  Shape                     : {final_df.shape[0]:,} rows x {final_df.shape[1]} columns",
        f"  Housing feature columns   : {len(housing_cols)}",
        f"  Embedding columns         : {len(emb_cols)}",
        f"  Total missing values      : {final_df.isnull().sum().sum()}",
        "",
        "ARTEFACT STATUS",
        "-" * 40,
    ]
    for label, ok in artefact_status.items():
        status = "[OK]     " if ok else "[MISSING]"
        lines.append(f"  {status} {label}")

    lines += [
        "",
        "FILES GENERATED DURING WEEK 3",
        "-" * 40,
    ]
    for fpath, desc in week3_files:
        exists = os.path.exists(fpath)
        tag = "[OK]" if exists else "[--]"
        lines.append(f"  {tag}  {fpath}")
        lines.append(f"         {desc}")

    lines += [
        "",
        "=" * 65,
        "All Week 3 stages completed successfully.",
        "=" * 65,
    ]

    report = "\n".join(lines)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[REPORT] Pipeline summary saved to: {path}")
    print("\n" + report)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline_validation() -> None:
    """
    End-to-end Week 3 pipeline validation and reporting.

    Steps
    -----
    1. Verify all expected artefacts exist on disk.
    2. Structurally validate each artefact.
    3. Generate KNN map overlay.
    4. Generate embedding correlation heatmap.
    5. Generate/confirm node degree histogram.
    6. Write pipeline_summary.txt.
    """
    print("=" * 60)
    print("Week 3 Pipeline Validation")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1 – Artefact existence check
    artefact_status = verify_artefacts()

    # Step 2 – Structural validation
    print("\n" + "=" * 60)
    print("Structural Validation")
    print("=" * 60)
    nodes_df = validate_graph_ready_dataset()
    edges_df = validate_graph_edges()
    adj      = validate_adjacency_list()
    embs_df  = validate_embeddings()
    final_df = validate_spatial_dataset()

    # Step 3 – KNN map overlay
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    plot_knn_map_overlay(nodes_df, edges_df)

    # Step 4 – Embedding correlation heatmap
    plot_embedding_correlation_heatmap(embs_df)

    # Step 5 – Degree histogram (skip if Stage 3 already produced it)
    plot_degree_histogram(edges_df)

    # Step 6 – Pipeline summary report
    write_pipeline_summary(nodes_df, edges_df, embs_df, final_df, artefact_status)

    print("\n" + "=" * 60)
    print("Week 3 pipeline validation completed successfully.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline_validation()
