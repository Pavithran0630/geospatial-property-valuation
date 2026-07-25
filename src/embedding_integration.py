"""
Embedding Integration Module
==============================

Week 3 – Stage 5: Merge the spatial node embeddings into the processed housing
dataset to produce the final Week 3 dataset.

Pipeline
--------
1. Load the graph-ready housing dataset (from Stage 1).
2. Load the spatial embeddings CSV (from Stage 4).
3. Merge both DataFrames on ``node_id`` (inner join).
4. Validate that no rows are lost during the merge.
5. Save the final dataset to ``data/spatial_dataset.csv``.
6. Print a structured summary:
   - embedding dimension
   - final dataset shape
   - missing-value report per column group

No models are retrained here.  The spatial_dataset.csv is the input for
Week 4 (GNN / Attention-Based modelling).
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NODES_PATH      = "data/graph_ready_data.csv"
DEFAULT_EMBEDDINGS_PATH = "data/spatial_embeddings.csv"
DEFAULT_OUTPUT_PATH     = "data/spatial_dataset.csv"

NODE_ID_COL = "node_id"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_housing_data(path: str = DEFAULT_NODES_PATH) -> pd.DataFrame:
    """
    Load the graph-ready housing dataset produced by Stage 1.

    Parameters
    ----------
    path : str
        Path to graph_ready_data.csv.

    Returns
    -------
    pandas.DataFrame
        Housing dataset with node_id as the first column.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Housing dataset not found at '{path}'. "
            "Please run src/graph_preprocessing.py first."
        )
    print(f"[INFO] Loading housing dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Housing dataset loaded — rows: {len(df):,}  |  columns: {df.shape[1]}")
    return df


def load_spatial_embeddings(path: str = DEFAULT_EMBEDDINGS_PATH) -> pd.DataFrame:
    """
    Load the spatial embeddings CSV produced by Stage 4.

    Parameters
    ----------
    path : str
        Path to spatial_embeddings.csv.

    Returns
    -------
    pandas.DataFrame
        Embedding matrix with node_id as the first column and
        emb_0 … emb_{dim-1} as subsequent columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Spatial embeddings not found at '{path}'. "
            "Please run src/spatial_embeddings.py first."
        )
    print(f"[INFO] Loading spatial embeddings from: {path}")
    df = pd.read_csv(path)
    embedding_dim = df.shape[1] - 1  # subtract node_id column
    print(f"[INFO] Embeddings loaded — nodes: {len(df):,}  |  embedding dim: {embedding_dim}")
    return df


def save_spatial_dataset(df: pd.DataFrame, path: str = DEFAULT_OUTPUT_PATH) -> None:
    """
    Save the merged spatial dataset to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        Final merged dataset.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Spatial dataset saved to: {path}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_node_id_column(df: pd.DataFrame, name: str) -> None:
    """
    Assert that the node_id column is present in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    name : str
        Human-readable name for error messages (e.g. "housing dataset").

    Raises
    ------
    KeyError
        If node_id is missing.
    """
    if NODE_ID_COL not in df.columns:
        raise KeyError(
            f"'{NODE_ID_COL}' column missing from {name}. "
            f"Available columns: {list(df.columns[:10])}"
        )
    print(f"[INFO] '{NODE_ID_COL}' column present in {name}. [OK]")


def validate_no_rows_lost(
    housing_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> None:
    """
    Ensure the merged dataset retains exactly as many rows as the housing
    dataset.  Any discrepancy means node_ids are misaligned between the two
    sources and must be investigated.

    Parameters
    ----------
    housing_df : pandas.DataFrame
        Original housing dataset (before merge).
    merged_df : pandas.DataFrame
        Dataset after merging embeddings.

    Raises
    ------
    ValueError
        If the merged row count differs from the original housing row count.
    """
    n_before = len(housing_df)
    n_after  = len(merged_df)

    if n_after != n_before:
        raise ValueError(
            f"Row count mismatch after merge! "
            f"Before: {n_before:,}  |  After: {n_after:,}  |  "
            f"Lost: {n_before - n_after:,} rows. "
            "Check that node_ids are consistent across both files."
        )
    print(f"[INFO] Row count validated — no rows lost ({n_after:,} rows). [OK]")


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_embeddings(
    housing_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the spatial embeddings into the housing dataset on ``node_id``.

    An inner join is used so that only nodes present in both DataFrames are
    retained.  The :func:`validate_no_rows_lost` check immediately after the
    merge ensures no properties are silently dropped.

    Parameters
    ----------
    housing_df : pandas.DataFrame
        Graph-ready housing dataset with node_id column.
    embeddings_df : pandas.DataFrame
        Spatial embeddings with node_id column.

    Returns
    -------
    pandas.DataFrame
        Merged dataset — all housing columns followed by embedding columns.
    """
    print(f"[INFO] Merging housing data ({len(housing_df):,} rows) "
          f"with embeddings ({len(embeddings_df):,} rows) on '{NODE_ID_COL}'...")

    merged_df = pd.merge(
        housing_df,
        embeddings_df,
        on=NODE_ID_COL,
        how="inner",
        validate="1:1",   # enforce one-to-one relationship on node_id
    )

    print(f"[INFO] Merge complete — resulting shape: {merged_df.shape}")
    return merged_df


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(
    housing_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> None:
    """
    Print a structured summary of the integration result.

    Covers:
    - Embedding dimension
    - Final dataset shape
    - Missing-value report broken down by column group
      (housing features vs. embedding dimensions)

    Parameters
    ----------
    housing_df : pandas.DataFrame
        Original housing dataset.
    embeddings_df : pandas.DataFrame
        Spatial embeddings DataFrame.
    merged_df : pandas.DataFrame
        Final merged dataset.
    """
    embedding_dim = embeddings_df.shape[1] - 1   # minus node_id col
    emb_cols      = [c for c in merged_df.columns if c.startswith("emb_")]
    housing_cols  = [c for c in merged_df.columns if c not in emb_cols]

    # Missing-value counts per group
    housing_nulls   = merged_df[housing_cols].isnull().sum().sum()
    embedding_nulls = merged_df[emb_cols].isnull().sum().sum()
    total_nulls     = housing_nulls + embedding_nulls

    print("\n" + "=" * 60)
    print("Week 3 - Final Spatial Dataset Summary")
    print("=" * 60)

    print(f"\n  Embedding dimension        : {embedding_dim}")
    print(f"  Final dataset shape        : {merged_df.shape[0]:,} rows x {merged_df.shape[1]} columns")
    print(f"    Housing feature cols     : {len(housing_cols)}")
    print(f"    Embedding dims (emb_*)   : {len(emb_cols)}")

    print(f"\n  Missing-Value Report")
    print(f"  {'-' * 40}")
    print(f"  Housing columns            : {housing_nulls} missing values")
    print(f"  Embedding columns          : {embedding_nulls} missing values")
    print(f"  Total missing values       : {total_nulls}")

    if total_nulls == 0:
        print("  [OK] Dataset is complete - no missing values detected.")
    else:
        # Report per-column breakdown for non-zero NaN columns
        null_by_col = merged_df.isnull().sum()
        problem_cols = null_by_col[null_by_col > 0]
        print(f"\n  Columns with missing values:")
        for col, cnt in problem_cols.items():
            print(f"    {col:<40}: {cnt:,}")

    print(f"\n  Output saved to            : {DEFAULT_OUTPUT_PATH}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_spatial_dataset(
    nodes_path:      str = DEFAULT_NODES_PATH,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    output_path:     str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    End-to-end pipeline that integrates spatial embeddings into the housing
    dataset to produce the final Week 3 dataset.

    Steps
    -----
    1. Load the graph-ready housing dataset.
    2. Load the spatial embeddings.
    3. Validate node_id presence in both DataFrames.
    4. Merge on node_id (inner join, 1-to-1 validated).
    5. Validate that no rows were lost.
    6. Save the merged dataset to data/spatial_dataset.csv.
    7. Print a structured summary report.

    Parameters
    ----------
    nodes_path : str
        Path to graph_ready_data.csv (Stage 1 output).
    embeddings_path : str
        Path to spatial_embeddings.csv (Stage 4 output).
    output_path : str
        Destination path for the final spatial_dataset.csv.

    Returns
    -------
    pandas.DataFrame
        The final merged spatial dataset.
    """
    print("=" * 60)
    print("Embedding Integration Pipeline")
    print("=" * 60)

    # Step 1 — Load housing dataset
    housing_df = load_housing_data(nodes_path)

    # Step 2 — Load spatial embeddings
    embeddings_df = load_spatial_embeddings(embeddings_path)

    # Step 3 — Validate node_id column exists in both DataFrames
    validate_node_id_column(housing_df, "housing dataset")
    validate_node_id_column(embeddings_df, "spatial embeddings")

    # Step 4 — Merge on node_id
    merged_df = merge_embeddings(housing_df, embeddings_df)

    # Step 5 — Validate no rows lost
    validate_no_rows_lost(housing_df, merged_df)

    # Step 6 — Save
    save_spatial_dataset(merged_df, output_path)

    # Step 7 — Summary
    print_summary(housing_df, embeddings_df, merged_df)

    print("\nEmbedding integration completed successfully.")

    return merged_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_spatial_dataset()
