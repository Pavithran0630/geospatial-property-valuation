"""
Graph Preprocessing Module
===========================

Week 3 – Stage 1: Prepare the processed housing dataset for graph construction.

This module performs the following steps:
1. Load the processed housing dataset produced by Week 1 / Week 2 pipelines.
2. Validate that latitude and longitude columns are present.
3. Remove rows that contain missing coordinate values.
4. Ensure latitude and longitude are stored as numeric (float64) types.
5. Reset the DataFrame index and assign a unique ``node_id`` to every property.
6. Save the graph-ready dataset to disk.

No graph is built here.  Graph construction is delegated to later stages.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column names used throughout this module
LAT_COL = "lat"
LON_COL = "long"
NODE_ID_COL = "node_id"

# Default I/O paths (relative to project root, matching Week 1/2 conventions)
DEFAULT_INPUT_PATH = "data/processed_house_data.csv"
DEFAULT_OUTPUT_PATH = "data/graph_ready_data.csv"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_processed_data(path: str = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """
    Load the processed housing dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the processed dataset CSV.
        Defaults to ``data/processed_house_data.csv``.

    Returns
    -------
    pandas.DataFrame
        Raw processed DataFrame as loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at '{path}'. "
            "Please run the Week 1/2 preprocessing pipeline first."
        )

    print(f"[INFO] Loading processed dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded  — rows: {df.shape[0]:,}  |  columns: {df.shape[1]}")
    return df


def save_graph_ready_data(df: pd.DataFrame, path: str = DEFAULT_OUTPUT_PATH) -> None:
    """
    Save the graph-ready DataFrame to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        Graph-ready dataset with ``node_id`` assigned.
    path : str
        Destination file path.
        Defaults to ``data/graph_ready_data.csv``.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Graph-ready dataset saved to: {path}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_coordinate_columns(df: pd.DataFrame) -> None:
    """
    Validate that the latitude and longitude columns are present in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to validate.

    Raises
    ------
    KeyError
        If either ``lat`` or ``long`` column is missing.
    """
    missing = [col for col in [LAT_COL, LON_COL] if col not in df.columns]
    if missing:
        raise KeyError(
            f"Required coordinate column(s) missing from dataset: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    print(f"[INFO] Coordinate columns validated — '{LAT_COL}' and '{LON_COL}' are present.")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def remove_missing_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where latitude or longitude values are missing (NaN).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (must already have lat/long columns).

    Returns
    -------
    pandas.DataFrame
        DataFrame with rows missing coordinates removed.
    """
    before = len(df)
    df = df.dropna(subset=[LAT_COL, LON_COL])
    removed = before - len(df)

    if removed > 0:
        print(f"[WARN] Removed {removed:,} row(s) with missing coordinates.")
    else:
        print("[INFO] No rows with missing coordinates found.")

    return df


def ensure_numeric_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce latitude and longitude columns to ``float64``.

    Rows where coercion fails (non-parseable strings) will have NaN introduced
    and are subsequently dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with lat/long columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with lat/long guaranteed to be ``float64``,
        and any un-coercible rows removed.
    """
    for col in [LAT_COL, LON_COL]:
        original_dtype = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if original_dtype != df[col].dtype:
            print(f"[INFO] Column '{col}' coerced from {original_dtype} to float64.")
        else:
            print(f"[INFO] Column '{col}' already numeric ({original_dtype}).")

    # Drop any rows that became NaN after coercion
    before = len(df)
    df = df.dropna(subset=[LAT_COL, LON_COL])
    coercion_drops = before - len(df)
    if coercion_drops > 0:
        print(
            f"[WARN] Dropped {coercion_drops:,} row(s) with non-numeric coordinate values."
        )

    return df


# ---------------------------------------------------------------------------
# Node ID assignment
# ---------------------------------------------------------------------------

def assign_node_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset the DataFrame index and assign a unique ``node_id`` to each property.

    The ``node_id`` is a zero-based integer that uniquely identifies each
    property node in the forthcoming graph structure.  It is inserted as the
    first column for easy reference.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned coordinate DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new leading ``node_id`` column
        (integer, 0-based, contiguous).
    """
    df = df.reset_index(drop=True)
    df.insert(0, NODE_ID_COL, df.index.astype(int))
    print(f"[INFO] Assigned node_id to {len(df):,} properties  (0 to {len(df) - 1}).")
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_graph_ready_dataset(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    End-to-end pipeline that prepares the processed housing dataset for graph
    construction.

    Steps
    -----
    1. Load the processed housing dataset.
    2. Validate latitude and longitude columns.
    3. Remove rows with missing coordinates.
    4. Ensure coordinates are numeric.
    5. Reset index and assign a unique ``node_id`` to every property.
    6. Save the graph-ready dataset to disk.

    Parameters
    ----------
    input_path : str
        Path to the processed dataset CSV.
    output_path : str
        Destination path for the graph-ready dataset CSV.

    Returns
    -------
    pandas.DataFrame
        The final graph-ready DataFrame.
    """
    print("=" * 60)
    print("Graph Preprocessing Pipeline")
    print("=" * 60)

    # Step 1 - Load
    df = load_processed_data(input_path)

    # Step 2 - Validate coordinate columns
    validate_coordinate_columns(df)

    # Step 3 - Remove missing coordinates
    df = remove_missing_coordinates(df)

    # Step 4 - Ensure coordinates are numeric
    df = ensure_numeric_coordinates(df)

    # Step 5 - Assign node IDs
    df = assign_node_ids(df)

    # Step 6 - Save
    save_graph_ready_data(df, output_path)

    # Summary
    print("\nGraph Preprocessing Summary")
    print("-" * 60)
    print(f"Total properties (nodes) : {len(df):,}")
    print(f"Total columns            : {df.shape[1]}")
    print(f"Coordinate columns       : '{LAT_COL}', '{LON_COL}'")
    print(f"Node ID column           : '{NODE_ID_COL}'")
    print(f"Lat range                : [{df[LAT_COL].min():.4f}, {df[LAT_COL].max():.4f}]")
    print(f"Lon range                : [{df[LON_COL].min():.4f}, {df[LON_COL].max():.4f}]")
    print(f"\nOutput saved to          : {output_path}")
    print("\nGraph preprocessing completed successfully.")
    print("=" * 60)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_graph_ready_dataset(
        input_path=DEFAULT_INPUT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
    )
