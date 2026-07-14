"""
Baseline Model Training

This module trains multiple regression models
for house price prediction and compares
their performance.
"""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
def load_data():

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")

    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }, predictions


def plot_predictions(y_test, predictions):

    plt.figure(figsize=(7,7))

    plt.scatter(y_test, predictions)

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title("Actual vs Predicted")

    plt.tight_layout()

    plt.savefig("results/prediction_vs_actual.png")

    plt.close()


def plot_residuals(y_test, predictions):

    residuals = y_test - predictions

    plt.figure(figsize=(7,5))

    plt.scatter(predictions, residuals)

    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig("results/residual_plot.png")

    plt.close()


def main():

    print("="*60)
    print("Baseline Model Training")
    print("="*60)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    X_train, X_test, y_train, y_test = load_data()
    
    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            )

    }

    results = []

    best_predictions = None
    best_model = None
    best_score = -999

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics, predictions = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(metrics)

        print(metrics)

        if metrics["R2 Score"] > best_score:

            best_score = metrics["R2 Score"]
            best_predictions = predictions
            best_model = model

        filename = name.lower().replace(" ", "_") + ".pkl"

        joblib.dump(
            model,
            f"models/{filename}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    plot_predictions(
        y_test,
        best_predictions
    )

    plot_residuals(
        y_test,
        best_predictions
    )

    print("\n")
    print(results_df)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()