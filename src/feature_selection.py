"""
Feature Selection & Data Preparation

This module prepares the processed housing dataset
for machine learning model training.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load processed dataset.
    """
    return pd.read_csv(path)


def drop_unnecessary_columns(df):
    """
    Remove columns that do not contribute
    to prediction.
    """

    columns_to_drop = ["id", "date"]

    return df.drop(columns=columns_to_drop)


def correlation_analysis(df):

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("results/correlation_heatmap.png")

    plt.close()

    print("Correlation heatmap saved.")


def feature_importance(df):

    X = df.drop(columns=["price"])

    y = df["price"]

    scores = mutual_info_regression(
        X,
        y,
        random_state=42
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": scores
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "results/feature_importance.csv",
        index=False
    )

    print("Feature importance saved.")

    return importance


def split_dataset(df):

    X = df.drop(columns=["price"])

    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42
    )

    X_train.to_csv(
        "data/X_train.csv",
        index=False
    )

    X_test.to_csv(
        "data/X_test.csv",
        index=False
    )

    y_train.to_csv(
        "data/y_train.csv",
        index=False
    )

    y_test.to_csv(
        "data/y_test.csv",
        index=False
    )

    print("Training and testing datasets saved.")


def main():

    print("=" * 60)
    print("Feature Selection Pipeline")
    print("=" * 60)

    print("\nLoading processed dataset...")

    df = load_data(
        "data/processed_house_data.csv"
    )

    print("Removing unnecessary columns...")

    df = drop_unnecessary_columns(df)

    print("Generating correlation heatmap...")

    correlation_analysis(df)

    print("Calculating feature importance...")

    importance = feature_importance(df)

    print("\nTop 10 Important Features")

    print(importance.head(10))

    print("\nSplitting dataset...")

    split_dataset(df)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()