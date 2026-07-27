import os
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==========================================================
# Create folders
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("Spatial Property Valuation - Final Model Training")
print("=" * 60)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/spatial_dataset.csv")

print(f"\nDataset Shape : {df.shape}")

# ==========================================================
# Prepare Features
# ==========================================================

target = "price"

drop_columns = [
    "price",
    "id",
    "date"
]

X = df.drop(columns=drop_columns)
y = df[target]

print(f"Number of Features : {X.shape[1]}")

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

# ==========================================================
# Models
# ==========================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
}

results = []

best_model = None
best_model_name = ""
best_r2 = -999

print("\nTraining Models...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    mape = np.mean(
        np.abs((y_test - predictions) / y_test)
    ) * 100

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4),
        "MAPE (%)": round(mape, 2)
    })

    print(
        f"{name:20} "
        f"R²={r2:.4f} "
        f"RMSE={rmse:.2f}"
    )

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name
# ==========================================================
# Save Model Comparison
# ==========================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

results_df.to_csv(
    "results/final_model_comparison.csv",
    index=False
)

print("\n")
print("=" * 60)
print("Model Comparison")
print("=" * 60)
print(results_df)

# ==========================================================
# Save Best Model
# ==========================================================

joblib.dump(
    best_model,
    "models/final_spatial_model.pkl"
)

# ==========================================================
# Save Best Model Metrics
# ==========================================================

best_metrics = results_df.iloc[[0]]
best_metrics.to_csv(
    "results/final_model_metrics.csv",
    index=False
)

print("\nBest Model :", best_model_name)
print("Best R²    :", round(best_r2, 4))

# ==========================================================
# Prediction vs Actual Plot
# ==========================================================

import matplotlib.pyplot as plt

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Prediction vs Actual ({best_model_name})")

plt.tight_layout()

plt.savefig(
    "results/final_prediction_vs_actual.png",
    dpi=300
)

plt.close()

# ==========================================================
# Feature Importance
# ==========================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "results/final_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 8))

    top = importance.head(20)

    plt.barh(
        top["Feature"],
        top["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title("Top 20 Important Features")

    plt.tight_layout()

    plt.savefig(
        "results/final_feature_importance.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Final Summary
# ==========================================================

print("\n")
print("=" * 60)
print("Training Completed Successfully")
print("=" * 60)

print(f"Best Model : {best_model_name}")
print(f"Best R²    : {best_r2:.4f}")

print("\nGenerated Files:")

print("models/final_spatial_model.pkl")
print("results/final_model_comparison.csv")
print("results/final_model_metrics.csv")
print("results/final_prediction_vs_actual.png")
print("results/final_feature_importance.csv")
print("results/final_feature_importance.png")

print("\nProject is ready for the Streamlit Dashboard.")