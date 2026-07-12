"""
Feature Engineering Module

This module creates new features from the cleaned housing dataset
to improve machine learning model performance.
"""

from datetime import datetime
import pandas as pd


def load_data(path):
    """
    Load cleaned housing dataset.

    Parameters
    ----------
    path : str
        Path to cleaned dataset.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(path)


def validate_columns(df):
    """
    Ensure all required columns exist.
    """

    required_columns = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def engineer_features(df):
    """
    Create engineered features.

    Features
    --------
    house_age
    renovation_age
    is_renovated
    total_rooms
    living_ratio
    basement_ratio

    Returns
    -------
    pandas.DataFrame
    """

    validate_columns(df)

    current_year = datetime.now().year

    # -----------------------------------
    # House Age
    # -----------------------------------
    df["house_age"] = current_year - df["yr_built"]

    # -----------------------------------
    # Renovation Age
    # -----------------------------------
    df["renovation_age"] = df["yr_renovated"].apply(
        lambda year: 0 if year == 0 else current_year - year
    )

    # -----------------------------------
    # Renovation Flag
    # -----------------------------------
    df["is_renovated"] = (
        df["yr_renovated"] > 0
    ).astype(int)

    # -----------------------------------
    # Total Rooms
    # -----------------------------------
    df["total_rooms"] = (
        df["bedrooms"] + df["bathrooms"]
    )

    # -----------------------------------
    # Living Area Ratio
    # -----------------------------------
    df["living_ratio"] = (
        df["sqft_living"] /
        df["sqft_lot"].replace(0, 1)
    )

    # -----------------------------------
    # Basement Ratio
    # -----------------------------------
    df["basement_ratio"] = (
        df["sqft_basement"] /
        df["sqft_living"].replace(0, 1)
    )

    return df


def save_data(df, path):
    """
    Save processed dataset.
    """

    df.to_csv(path, index=False)


def main():
    """
    Execute feature engineering pipeline.
    """

    print("=" * 55)
    print("Feature Engineering Pipeline")
    print("=" * 55)

    print("\nLoading cleaned dataset...")

    df = load_data("data/cleaned_house_data.csv")

    print("Creating engineered features...")

    df = engineer_features(df)

    print("Saving processed dataset...")

    save_data(df, "data/processed_house_data.csv")

    print("\nFeature Engineering Summary")
    print("-" * 55)

    print(f"Rows      : {df.shape[0]}")
    print(f"Columns   : {df.shape[1]}")

    print("\nNew Features Created")

    new_features = [
        "house_age",
        "renovation_age",
        "is_renovated",
        "total_rooms",
        "living_ratio",
        "basement_ratio",
    ]

    for feature in new_features:
        print(f"✓ {feature}")

    print("\nProcessed dataset saved to:")
    print("data/processed_house_data.csv")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()