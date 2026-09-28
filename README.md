# Load-Shedding ML Prediction

Machine learning system for predicting load-shedding risk in South Africa, using historical Eskom system data and hourly weather observations.

## Overview

End-to-end pipeline following CRISP-DM: data loading, feature engineering, model training and comparison, evaluation, and deployment artifact generation. The final deployment model is a binary HistGradientBoosting classifier that predicts whether load-shedding will occur in the next hour.

**Models compared:** Random Forest, XGBoost, HistGradientBoosting (multiclass), HistGradientBoosting (binary deployment), and an LSTM for comparison.

## Data

| Source | Description |
|---|---|
| `ESK2033.csv` | Hourly Eskom system-status export (2018–2023) |
| `EskomSePush_history.csv` | EskomSePush load-shedding stage history |
| Open-Meteo Archive API | Hourly weather for the Free State reference location |

## Pipeline

1. **Phase 1 — Data Loading:** normalise column names, merge Eskom + weather + EskomSePush into one hourly dataset.
2. **Phase 2 — Feature Engineering:** time features, 1/2/24-hour lags, 6/24-hour rolling statistics. Train/test split is time-based (80/20) with `stage_lag1/2/24` explicitly excluded to prevent target leakage.
3. **Phase 3 — Model Training:** train multiclass models, select the best by weighted F1, train the binary deployment model, tune the decision threshold via precision-recall.
4. **Phase 4 — Persistence:** save the model, scaler, threshold, and metrics; write historical and prediction tables to SQLite.
5. **Phase 5 — Deployment:** Flask REST API + interactive Dash dashboard.

## Deployment Model Performance

| Metric | Value |
|---|---|
| Accuracy | 0.8972 |
| F1 (weighted) | 0.8972 |
| ROC-AUC | 0.9345 |
| Decision threshold | 0.0296 |

## Running the Dashboard

```bash
pip install -r requirements.txt
python dash_app.py
```

Then open [http://localhost:8050](http://localhost:8050).

## Repository Contents

| File | Purpose |
|---|---|
| `Mutavhatsindi Thendo Elrich Capstone 202321038.ipynb` | Full end-to-end pipeline notebook |
| `dash_app.py` | Interactive Dash dashboard |
| `api.py` | Flask REST API |
| `final_model.pkl` | Trained binary HistGradientBoosting model |
| `scaler.pkl` | Fitted StandardScaler |
| `feature_names.json` | Ordered list of 97 model features |
| `threshold.json` | F1-optimal decision threshold |
| `model_metrics.json` | Evaluation metrics for all models |
| `feature_importances.csv` | Ranked feature importances |
| `figures/` | Diagnostic plots (feature importance, confusion matrix, PR curve) |

## Author

Thendo Elrich Mutavhatsindi — 202321038
