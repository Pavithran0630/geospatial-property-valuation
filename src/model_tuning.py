"""
Model Tuning using GridSearchCV

This module tunes the Random Forest model,
evaluates its performance, and saves the
best model.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


def load_data():

    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")

    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def tune_model(X_train, y_train):

    param_grid = {

        "n_estimators": [100, 200],

        "max_depth": [10, 20, None],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]

    }

    grid = GridSearchCV(

        estimator=RandomForestRegressor(
            random_state=42
        ),

        param_grid=param_grid,

        cv=5,

        scoring="r2",

        n_jobs=-1

    )

    grid.fit(X_train, y_train)

    return grid


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    mape = mean_absolute_percentage_error(
        y_test,
        predictions
    ) * 100

    return mae, rmse, r2, mape


def save_results(best_params, metrics):

    os.makedirs("results", exist_ok=True)

    os.makedirs("models", exist_ok=True)

    pd.DataFrame([best_params]).to_csv(

        "results/best_parameters.csv",

        index=False

    )

    metrics.to_csv(

        "results/optimized_model_metrics.csv",

        index=False

    )


def main():

    print("=" * 60)

    print("Random Forest Hyperparameter Tuning")

    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    print("\nSearching best parameters...\n")

    grid = tune_model(X_train, y_train)

    print("Best Parameters")

    print(grid.best_params_)

    best_model = grid.best_estimator_

    mae, rmse, r2, mape = evaluate(

        best_model,

        X_test,

        y_test

    )

    metrics = pd.DataFrame({

        "MAE": [mae],

        "RMSE": [rmse],

        "R2 Score": [r2],

        "MAPE (%)": [mape]

    })

    print("\nOptimized Model Performance\n")

    print(metrics)

    joblib.dump(

        best_model,

        "models/best_random_forest.pkl"

    )

    save_results(

        grid.best_params_,

        metrics

    )

    print("\nBest model saved successfully!")

    print("Results saved successfully!")


if __name__ == "__main__":

    main()