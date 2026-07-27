# 🏡 Geospatial Property Valuation using Spatial Embeddings

An end-to-end Machine Learning project that predicts residential property prices using **graph-based spatial embeddings (Node2Vec)** and **XGBoost Regression**. The project combines traditional housing features with geographic relationships to improve prediction accuracy.

---

## 📌 Project Overview

Traditional house price prediction models rely mainly on property attributes such as living area, bedrooms, and bathrooms.

This project enhances prediction performance by incorporating **spatial information**. Nearby houses are represented as a graph, Node2Vec generates spatial embeddings, and XGBoost learns both property characteristics and neighborhood relationships.

The final model achieved an **R² Score of 0.8732**.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Spatial Graph Construction
- Node2Vec Spatial Embeddings
- Multiple Regression Models
- XGBoost Model Selection
- SHAP Explainability
- Interactive Streamlit Dashboard
- House Price Prediction Interface

---

## 🛠 Technology Stack

| Category | Tools |
|----------|------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Spatial Analysis | GeoPandas, NetworkX, Node2Vec |
| Deployment | Streamlit |

---

## 📂 Project Structure

```text
geospatial-property-valuation/
│
├── dashboard/
│   └── app.py
│
├── data/
├── models/
├── notebooks/
├── results/
├── src/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## ⚙️ Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Spatial Graph Construction
      │
      ▼
Node2Vec Embeddings
      │
      ▼
XGBoost Regression
      │
      ▼
Model Evaluation
      │
      ▼
SHAP Explainability
      │
      ▼
Streamlit Dashboard
```

---

## 📊 Model Performance

| Model | R² Score |
|------|---------:|
| XGBoost | **0.8732** |
| Random Forest | 0.8507 |
| Gradient Boosting | 0.8450 |
| Linear Regression | 0.7079 |

### Best Model

- **Model:** XGBoost
- **R² Score:** 0.8732
- **Lowest RMSE**
- **Production Model Selected**

---

## 📈 Dashboard Features

- 🏠 Home Dashboard
- 📊 Dataset Overview
- 🤖 Model Performance
- ⭐ SHAP Explainability
- 🌍 Spatial Analysis
- 💰 Interactive House Price Prediction

---

## 📷 Dashboard Preview

> Add screenshots here after capturing them.

```
assets/
├── home.png
├── dataset.png
├── model.png
├── explainability.png
├── spatial.png
└── prediction.png
```

---

## ▶️ Installation

```bash
git clone https://github.com/Pavithran0630/geospatial-property-valuation.git

cd geospatial-property-valuation

pip install -r requirements.txt

streamlit run dashboard/app.py
```

---

## 📌 Future Improvements

- Deep Learning models
- Real-time property data
- Cloud Deployment
- API Integration
- Time-series house price forecasting

---

## 👨‍💻 Authors

**Pavithran M**
**Charul Thakur**
**Praveen Nandan**

