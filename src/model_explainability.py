import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Create results folder
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("SHAP Model Explainability")
print("=" * 60)

# Load dataset
df = pd.read_csv("data/spatial_dataset.csv")

X = df.drop(columns=["price", "id", "date"])
y = df["price"]

# Load trained model
model = joblib.load("models/final_spatial_model.pkl")

print("Generating SHAP values...")

# Use only first 500 samples for speed
sample = X.sample(500, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Summary Plot
plt.figure(figsize=(12,8))

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Bar Plot
plt.figure(figsize=(10,8))

shap.summary_plot(
    shap_values,
    sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGenerated Files")
print("----------------")
print("results/shap_summary.png")
print("results/shap_bar.png")

print("\nDone!")