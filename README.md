# Diabetes Prediction and Clustering (CDC Dataset) — binary_drop1

This project uses the **CDC Diabetes dataset** to do two tasks:

1) **Classification (prediction):** predict if a person is diabetic or not.
2) **Clustering (grouping):** group people into clusters based on similar health patterns.

The code saves results automatically as **figures**, **reports**, and the **best model**.

---

## What dataset is used?

- File: `CDC Diabetes Dataset.csv`
- Target column: `Diabetes_012`

The original target values are:
- 0 = no diabetes
- 1 = prediabetes
- 2 = diabetes

### Target used in this project (binary_drop1)
This project uses **binary_drop1**:

- Keep only rows where `Diabetes_012` is **0 or 2**
- Remove rows where `Diabetes_012` is **1** (prediabetes)
- Convert `2 -> 1`

Final meaning:
- 0 = non-diabetic
- 1 = diabetic

After filtering:
- Rows used: **249,049**
- Class counts:
  - 0 (non-diabetic): **213,703**
  - 1 (diabetic): **35,346**

---

## What models are used?

### Classification models
- **DummyMostFreq** (baseline model)
- **Logistic Regression**
- **Random Forest** (best model in this run)

Best model saved:
- Best model: **RandomForest**
- Best test Macro-F1: **0.6750**
- Saved file: `outputs/models/best_model_RandomForest.joblib`

### Clustering model
- **MiniBatchKMeans** (fast clustering for large datasets)

Clustering results (from this run):
- Best k: **2**
- Silhouette (sample-based): **0.2219**
- Davies–Bouldin (full): **2.4564**
- Calinski–Harabasz (full): **31640.39**

Cluster vs target distribution (important result):
- Cluster 0: diabetic = **0.0879**
- Cluster 1: diabetic = **0.3249**
So **Cluster 1 has higher diabetes proportion**.

---
## How to run 
### 1) Create and activate a virtual environment (recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```
## Install packages 
```
pip install -r requirements.txt
```
## place the dataset
### put the csv here :
```
data/raw/CDC Diabetes Dataset.csv

```
## Run the project 
### From the project root 
```
python run_pipeline.py
```

