# 🏠 Geospatial Property Valuation using Spatial Embeddings

## 📌 Project Overview

This project focuses on predicting real estate prices using geospatial information. Traditional valuation models rely only on tabular features such as bedrooms, bathrooms, and square footage. This project extends the valuation process by incorporating spatial relationships between neighboring properties, enabling more accurate property price estimation.

The project follows a four-week development roadmap, beginning with geospatial data preprocessing and ending with a Graph Neural Network (GNN) based valuation model and an interactive dashboard.

---

## 📂 Dataset

- Dataset: King County House Sales Dataset
- Features:
  - Price
  - Bedrooms
  - Bathrooms
  - Living Area
  - Lot Area
  - Floors
  - Latitude
  - Longitude
  - Year Built
  - Waterfront
  - Condition
  - Grade

---

# Week 1 – Geospatial Data Acquisition & Processing

## Objectives

- Load and inspect the dataset
- Perform Exploratory Data Analysis (EDA)
- Handle missing values
- Remove duplicate records
- Detect and remove outliers
- Create GeoDataFrame using GeoPandas
- Visualize properties on an interactive Folium map

---

## Technologies Used

- Python
- Pandas
- NumPy
- GeoPandas
- Shapely
- Folium
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

```
geospatial-property-valuation/

│── data/
│── notebooks/
│── src/
│── results/
│── dashboard/
│── models/
│── README.md
│── requirements.txt
```

---

## Outputs

- Cleaned Housing Dataset
- Interactive Property Map
- Price Distribution Analysis
- Correlation Analysis

---

## Next Steps

Week 2:
- Feature Engineering
- Linear Regression
- XGBoost Baseline
- RMSE & MAPE Evaluation

## Team Members

- Pavitharan
- Charul
- Praveen Nandan