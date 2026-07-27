import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
import matplotlib.pyplot as plt
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Geospatial Property Valuation",
    page_icon="🏡",
    layout="wide"
)
st.markdown("""
<style>
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

div[data-testid="metric-container"]{
    border:1px solid #E6E6E6;
    border-radius:12px;
    padding:18px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

h1{
    color:#1f77b4;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🏡 Geospatial Property Valuation using Spatial Embeddings")

st.markdown("""
This dashboard presents the complete machine learning pipeline for predicting
house prices using **geospatial features**, **graph-based spatial embeddings (Node2Vec)**,
and **XGBoost Regression**.
""")

# -----------------------------
# Sidebar
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "🤖 Model Performance",
        "⭐ Explainability",
        "📈 Predictions",
        "🌍 Spatial Analysis",
        "💰 Predict House Price"
    ]
)

# -----------------------------
# HOME
# -----------------------------
if page == "🏠 Home":

    st.title("🏡 Geospatial Property Valuation")
    st.caption("Predicting House Prices using Spatial Embeddings and Machine Learning")

    st.markdown("---")

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🏠 Houses", "20,467")
    c2.metric("📊 Features", "89")
    c3.metric("🤖 Best Model", "XGBoost")
    c4.metric("🎯 R² Score", "0.8732")

    st.markdown("---")

    st.header("📌 Project Objective")

    st.write("""
This project predicts residential property prices by combining
traditional house features with **graph-based spatial embeddings**.

Instead of relying only on property characteristics, nearby houses
are represented as a graph, allowing the model to learn geographical
relationships and improve prediction accuracy.
""")

    st.markdown("---")

    st.header("⚙️ Project Workflow")

    st.info("""
📂 Raw Dataset

⬇️

🧹 Data Cleaning & Preprocessing

⬇️

⚙️ Feature Engineering

⬇️

🗺️ Spatial Graph Construction

⬇️

🧠 Node2Vec Spatial Embeddings

⬇️

🤖 XGBoost Regression

⬇️

📈 Model Evaluation

⬇️

⭐ SHAP Explainability

⬇️

🌐 Interactive Streamlit Dashboard
""")

    st.markdown("---")

    st.header("🛠 Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.success("""
Python

Pandas

NumPy

Scikit-Learn
""")

    with tech2:
        st.success("""
XGBoost

SHAP

NetworkX

Node2Vec
""")

    with tech3:
        st.success("""
GeoPandas

Matplotlib

Streamlit

Joblib
""")
# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.header("📊 Dataset Overview")

    df = pd.read_csv("data/spatial_dataset.csv")

    # ==========================
    # KPI Cards
    # ==========================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Target", "Price")

    st.markdown("---")

    # ==========================
    # Dataset Preview
    # ==========================
    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # Price Statistics
    # ==========================
    st.subheader("Price Statistics")

    stats = pd.DataFrame(df["price"].describe())

    st.dataframe(
        stats,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # Price Distribution
    # ==========================
    st.subheader("House Price Distribution")

    import plotly.express as px

    fig = px.histogram(
        df,
        x="price",
        nbins=40,
        title="Distribution of House Prices"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # Correlation Heatmap
    # ==========================
    st.subheader("Top Feature Correlations")

    numeric_df = df.select_dtypes(include="number")

    corr = numeric_df.corr()["price"] \
                    .abs() \
                    .sort_values(ascending=False) \
                    .head(10)

    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", "Correlation"]

    fig = px.bar(
        corr_df,
        x="Feature",
        y="Correlation",
        title="Top Features Correlated with Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # Map
    # ==========================
    if "lat" in df.columns and "long" in df.columns:

        st.subheader("Property Locations")

        map_df = df[["lat", "long"]].rename(
            columns={
                "lat": "lat",
                "long": "lon"
            }
        )

        st.map(
            map_df.sample(min(1000, len(map_df)))
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.header("🤖 Model Performance Dashboard")

    comparison = pd.read_csv("results/final_model_comparison.csv")

    # -----------------------------
    # Best Model Card
    # -----------------------------
    best_model = comparison.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("🏆 Best Model", best_model["Model"])
    c2.metric("🎯 R² Score", f"{best_model['R2 Score']:.4f}")
    c3.metric("📉 RMSE", f"{best_model['RMSE']:,.0f}")

    st.markdown("---")

    # -----------------------------
    # Leaderboard
    # -----------------------------
    st.subheader("🏅 Model Leaderboard")

    st.dataframe(
        comparison.style.highlight_max(
            subset=["R2 Score"],
            color="#90EE90"
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # R² Comparison
    # -----------------------------
    st.subheader("📊 Model Comparison")

    import plotly.express as px

    fig = px.bar(
        comparison,
        x="Model",
        y="R2 Score",
        color="R2 Score",
        text="R2 Score",
        title="R² Score Comparison"
    )

    fig.update_traces(texttemplate="%{text:.4f}")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # Error Comparison
    # -----------------------------
    fig = px.bar(
        comparison,
        x="Model",
        y="RMSE",
        color="RMSE",
        title="RMSE Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # Prediction Plot
    # -----------------------------
    st.subheader("📈 Actual vs Predicted")

    if os.path.exists("results/final_prediction_vs_actual.png"):
        st.image(
            "results/final_prediction_vs_actual.png",
            use_container_width=True
        )

    st.markdown("---")

    # -----------------------------
    # Performance Summary
    # -----------------------------
    st.success("""
### ✔ Final Result

• Best Performing Model: **XGBoost**

• R² Score: **0.8732**

• Lowest Prediction Error

• Selected as Final Production Model
""")
# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "⭐ Explainability":

    st.header("⭐ Model Explainability")

    st.write("""
This section explains how the XGBoost model makes predictions using SHAP values
and feature importance analysis.
""")

    st.markdown("---")

    # ==========================
    # SHAP Summary
    # ==========================

    st.subheader("🔍 SHAP Summary Plot")

    if os.path.exists("results/shap_summary.png"):
        st.image(
            "results/shap_summary.png",
            use_container_width=True
        )

    st.info("""
SHAP values measure how much each feature contributes to a prediction.

• Positive SHAP values increase the predicted price.

• Negative SHAP values decrease the predicted price.

• Color represents feature value (Red = High, Blue = Low).
""")

    st.markdown("---")

    # ==========================
    # Feature Importance
    # ==========================

    st.subheader("📊 Global Feature Importance")

    if os.path.exists("results/final_feature_importance.png"):
        st.image(
            "results/final_feature_importance.png",
            use_container_width=True
        )

    st.markdown("---")

    # ==========================
    # Top Features Table
    # ==========================

    if os.path.exists("results/final_feature_importance.csv"):

        importance = pd.read_csv(
            "results/final_feature_importance.csv"
        )

        st.subheader("🏆 Top 20 Most Important Features")

        st.dataframe(
            importance.head(20),
            use_container_width=True
        )

    st.markdown("---")

    # ==========================
    # Interactive Importance Chart
    # ==========================

    import plotly.express as px

    fig = px.bar(
        importance.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Important Features",
        color="Importance"
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # Key Insights
    # ==========================

    st.success("""
### 💡 Key Insights

• Spatial embeddings significantly improve prediction performance.

• Living area and house quality are among the strongest predictors.

• The XGBoost model captures complex non-linear relationships between features.

• SHAP values provide transparent explanations for every prediction.
""")


# ============================================================
# PREDICTIONS
# ============================================================

elif page == "📈 Predictions":

    st.header("📈 Prediction Results")

    if os.path.exists("results/final_prediction_vs_actual.png"):

        st.image(
            "results/final_prediction_vs_actual.png",
            use_container_width=True
        )

    st.markdown("---")

    st.write("""
    The scatter plot compares the actual house prices with the model's
    predicted prices.

    • Points close to the diagonal indicate accurate predictions.

    • A high R² score (0.8732) shows that the model captures most of the
      variation in house prices.
    """)


# ============================================================
# SPATIAL ANALYSIS
# ============================================================

elif page == "🌍 Spatial Analysis":

    st.header("🌍 Spatial Analysis")

    st.write("""
This project leverages geographic information to improve house price prediction.
Node2Vec spatial embeddings capture relationships between neighboring properties,
allowing the XGBoost model to learn location-aware patterns.
""")

    df = pd.read_csv("data/spatial_dataset.csv")

    import plotly.express as px

    # =====================================================
    # Map
    # =====================================================

    if {"lat", "long", "price"}.issubset(df.columns):

        st.subheader("📍 Property Locations")

        sample_df = df.sample(min(2000, len(df)), random_state=42)

        fig = px.scatter_mapbox(
            sample_df,
            lat="lat",
            lon="long",
            color="price",
            size="price",
            hover_data=["price"],
            zoom=8,
            height=600,
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =====================================================
    # Latitude vs Longitude
    # =====================================================

    st.subheader("📊 Geographic Distribution")

    fig = px.scatter(
        df.sample(min(5000, len(df)), random_state=42),
        x="long",
        y="lat",
        color="price",
        opacity=0.7,
        title="House Locations Colored by Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =====================================================
    # Price Distribution
    # =====================================================

    st.subheader("💲 Price Distribution")

    fig = px.box(
        df,
        y="price",
        title="House Price Spread"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =====================================================
    # Statistics
    # =====================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Price",
        f"${df['price'].mean():,.0f}"
    )

    c2.metric(
        "Median Price",
        f"${df['price'].median():,.0f}"
    )

    c3.metric(
        "Maximum Price",
        f"${df['price'].max():,.0f}"
    )

    st.markdown("---")

    st.success("""
### 🌎 Spatial Insights

• Nearby properties often exhibit similar prices.

• Geographic information significantly improves prediction accuracy.

• Node2Vec embeddings encode neighborhood relationships into numerical features.

• The model learns both property characteristics and spatial context.
""")
# ============================================================
# HOUSE PRICE PREDICTION
# ============================================================

elif page == "💰 Predict House Price":

    st.header("🏡 House Price Prediction")
    st.write("Adjust the property details below and estimate the house price using the trained XGBoost model.")

    df = pd.read_csv("data/spatial_dataset.csv")
    model = joblib.load("models/final_spatial_model.pkl")

    X = df.drop(columns=["price", "id", "date"])
    input_data = X.mean().to_dict()

    st.markdown("### Property Details")

    col1, col2 = st.columns(2)

    with col1:

        if "bedrooms" in input_data:
            input_data["bedrooms"] = st.slider(
                "Bedrooms",
                1, 10,
                int(input_data["bedrooms"])
            )

        if "bathrooms" in input_data:
            input_data["bathrooms"] = st.slider(
                "Bathrooms",
                1.0, 8.0,
                float(input_data["bathrooms"]),
                step=0.5
            )

        if "floors" in input_data:
            input_data["floors"] = st.slider(
                "Floors",
                1.0, 4.0,
                float(input_data["floors"]),
                step=0.5
            )

    with col2:

        if "sqft_living" in input_data:
            input_data["sqft_living"] = st.number_input(
                "Living Area (sqft)",
                300,
                10000,
                int(input_data["sqft_living"])
            )

        if "grade" in input_data:
            input_data["grade"] = st.slider(
                "Grade",
                1, 13,
                int(input_data["grade"])
            )

        if "condition" in input_data:
            input_data["condition"] = st.slider(
                "Condition",
                1, 5,
                int(input_data["condition"])
            )

    st.markdown("---")

    if st.button("🔮 Predict House Price", use_container_width=True):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        st.balloons()

        st.success("## 🎉 Prediction Complete!")

        st.metric(
            label="Estimated House Price",
            value=f"${prediction:,.2f}"
        )

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        c1.metric("Model", "XGBoost")
        c2.metric("R² Score", "0.8732")
        c3.metric("Features Used", len(input_df.columns))

        st.markdown("---")

        st.subheader("Input Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Bedrooms",
                "Bathrooms",
                "Living Area",
                "Floors",
                "Grade",
                "Condition"
            ],
            "Value": [
                input_data.get("bedrooms", "-"),
                input_data.get("bathrooms", "-"),
                input_data.get("sqft_living", "-"),
                input_data.get("floors", "-"),
                input_data.get("grade", "-"),
                input_data.get("condition", "-")
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.info("""
Prediction generated using the trained **XGBoost Regression** model with
engineered features and spatial embeddings. The remaining features are
automatically filled with their dataset mean values.
""")