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

---

# Week 3 – Spatial Embeddings & Graph Construction

## Overview

Week 3 extends the project beyond tabular ML by converting the housing dataset
into a **graph** where every property is a node and edges connect spatially
nearby properties.  Node2Vec random walks are then used to learn **spatial
embeddings** that encode each property's neighbourhood context into a dense
vector.  These embeddings are merged back into the housing dataset to create
the final `spatial_dataset.csv` which feeds into the Week 4 GNN model.

---

## Graph Construction

Each of the 20,467 King County properties becomes a **node** in a directed
**K-Nearest Neighbours (KNN) graph** (default k = 5).

- Spatial distances are computed using the **Haversine formula** on lat/long.
- A **BallTree** index (sklearn) provides efficient nearest-neighbour lookup.
- Two output structures are produced:
  - **Edge list** – every directed connection as `(source, target, distance_km)`.
  - **Adjacency list** – `{node_id: [neighbour_ids]}` dict (pickle).

### Graph Statistics

| Metric | Value |
|---|---|
| Total nodes | 20,467 |
| Total directed edges | 102,335 |
| Neighbours per node (k) | 5 |
| Avg neighbour distance | 0.20 km |
| Max neighbour distance | 23.52 km |
| Weakly connected components | 41 |

---

## KNN Neighbourhoods

The graph captures the idea that **a house's value is influenced by its
immediate spatial neighbours**.  With k = 5, each property is connected to
its 5 geographically closest properties.  Graph analysis (Stage 3) confirmed:

- The graph is highly **sparse** (density ≈ 0.000244).
- The largest weakly connected component covers 4,980 nodes.
- All nodes have exactly out-degree 5 (by construction of KNN).

---

## Spatial Embeddings

**Method: Node2Vec** (biased random walks + Word2Vec skip-gram)

Node2Vec learns a 64-dimensional vector for each property node by:
1. Simulating 10 random walks of length 30 per node.
2. Feeding the walk sequences as "sentences" into a Word2Vec model.
3. The resulting vectors capture which properties appear in similar
   neighbourhood contexts.

**Fallback:** If Node2Vec is unavailable, a neighbourhood mean-pooling + PCA
approach is used automatically.

### Embedding Parameters (defaults)

| Parameter | Value |
|---|---|
| Embedding dimension | 64 |
| Walk length | 30 |
| Walks per node | 10 |
| p (return) | 1.0 |
| q (in-out) | 1.0 |
| Word2Vec window | 5 |
| Training epochs | 5 |

---

## Generated Outputs

### Data Files

| File | Description |
|---|---|
| `data/graph_ready_data.csv` | Cleaned dataset with `node_id` (Stage 1) |
| `data/graph_edges.csv` | Edge list: source, target, distance_km (Stage 2) |
| `data/adjacency.pkl` | Adjacency dict: node_id → [neighbours] (Stage 2) |
| `data/spatial_embeddings.csv` | 20,467 × 65 Node2Vec embeddings (Stage 4) |
| `data/spatial_dataset.csv` | Final dataset: 20,467 × 92 (housing + embeddings) (Stage 5) |

### Result Files

| File | Description |
|---|---|
| `results/graph_statistics.csv` | Key graph metrics table (Stage 3) |
| `results/degree_distribution.png` | Node degree distribution histogram (Stage 3) |
| `results/graph_visualization.png` | Spatial subgraph scatter plot (Stage 3) |
| `results/knn_map_overlay.png` | KNN connections overlaid on property map (Stage 6) |
| `results/embedding_correlation_heatmap.png` | Embedding dimension correlations (Stage 6) |
| `results/pipeline_summary.txt` | Full Week 3 validation report (Stage 6) |

---

## Project Structure (Week 3 additions)

```
geospatial-property-valuation/
│
├── data/
│   ├── graph_ready_data.csv        # Stage 1 output
│   ├── graph_edges.csv             # Stage 2 output
│   ├── adjacency.pkl               # Stage 2 output
│   ├── spatial_embeddings.csv      # Stage 4 output
│   └── spatial_dataset.csv         # Stage 5 output (final Week 3 dataset)
│
├── src/
│   ├── graph_preprocessing.py      # Stage 1 – coordinate validation + node_id
│   ├── graph_builder.py            # Stage 2 – KNN graph construction
│   ├── graph_analysis.py           # Stage 3 – graph metrics + visualizations
│   ├── spatial_embeddings.py       # Stage 4 – Node2Vec spatial embeddings
│   ├── embedding_integration.py    # Stage 5 – merge embeddings into dataset
│   └── pipeline_validation.py      # Stage 6 – end-to-end validation + report
│
└── results/
    ├── graph_statistics.csv
    ├── degree_distribution.png
    ├── graph_visualization.png
    ├── knn_map_overlay.png
    ├── embedding_correlation_heatmap.png
    └── pipeline_summary.txt
```

---

## How to Run Week 3

Run each stage in order from the project root using the venv Python:

```bash
# Stage 1 – Prepare graph-ready dataset
python src/graph_preprocessing.py

# Stage 2 – Build KNN graph
python src/graph_builder.py

# Stage 3 – Analyse graph
python src/graph_analysis.py

# Stage 4 – Generate Node2Vec spatial embeddings
python src/spatial_embeddings.py

# Stage 5 – Merge embeddings into final dataset
python src/embedding_integration.py

# Stage 6 – Validate full pipeline and generate report
python src/pipeline_validation.py
```

To change the embedding dimension (default 64), edit `spatial_embeddings.py`:

```python
# In src/spatial_embeddings.py
DEFAULT_EMBEDDING_DIM = 128   # or any value
```

To change the number of KNN neighbours (default k = 5), edit `graph_builder.py`:

```python
# In src/graph_builder.py
DEFAULT_K = 10   # or any value
```

---

## Next Steps – Week 4

- Train a **Graph Neural Network (GNN)** on the spatial dataset.
- Implement an **Attention-Based Spatial Model** to weight neighbours.
- Compare GNN MAPE vs XGBoost baseline.
- Deploy a **Streamlit dashboard** with predicted price heatmaps.