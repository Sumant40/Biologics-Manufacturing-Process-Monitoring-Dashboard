# Biologics Manufacturing Process Monitoring Dashboard

A end-to-end **Multivariate Statistical Process Control (MSPC)** system for bioreactor batch monitoring,
built to mirror real workflows used in drug substance process development (PANDAS-style roles at companies like Amgen, Genentech, and BMS).

---

## What This Project Does

In biologics manufacturing, a single failed batch can cost millions of dollars and delay patient access to life-saving drugs. Process engineers and data scientists on PANDAS (Process ANalytics and Data Science) teams monitor hundreds of parameters across 14-day bioreactor runs to catch problems early.

This dashboard replicates that workflow end-to-end:

1. **Simulates** realistic CHO (Chinese Hamster Ovary) cell culture data across 80 batches — the same cell line used to manufacture drugs like Herceptin and Avastin.
2. **Stores** all batch data in a structured SQL database, mirroring how manufacturing execution systems (MES) persist process records.
3. **Applies MSPC** using PCA + Hotelling's T² to flag out-of-control batches — the industry-standard approach described in ICH Q10 and FDA process validation guidelines.
4. **Predicts final yield** from only the first 48 hours of a run using XGBoost, giving engineers an early warning signal before a batch fails.
5. **Visualizes everything** in an interactive Plotly Dash dashboard with real-time control charts, trend plots, and yield predictions.

---

## Why MSPC Instead of Simple Univariate Control Charts?

Traditional SPC watches one variable at a time (e.g. just pH, or just temperature). But bioprocesses are highly multivariate — pH, dissolved oxygen, cell density, and feed rates are all correlated. A problem often shows up as a subtle *pattern* across multiple parameters, invisible to any single chart.

**Multivariate SPC** captures the full covariance structure:

| Method | What it detects |
|---|---|
| **PCA** | Compresses 10 correlated parameters into 5 independent principal components |
| **Hotelling's T²** | Flags batches that deviate from the normal operating region *within* the PCA model |
| **SPE / Q-statistic** | Flags batches behaving in a structurally new way *outside* the PCA model |

Together, T² and SPE give complete coverage:

| | T² Normal | T² High |
|---|---|---|
| **SPE Normal** | ✅ In control | Unusual direction, known failure mode |
| **SPE High** | New type of variation | 🔴 Total process upset |

---

## Yield Prediction: Why 48 Hours?

In a 14-day fed-batch culture, the first 2 days establish the trajectory. Early cell growth rate, pH stability, and dissolved oxygen response are strong predictors of final titer. By predicting yield at 48h, a process engineer can:

- **Intervene early** (adjust feeds, correct pH drift) before the batch is lost
- **Prioritize downstream resources** — manufacturing suites book purification slots weeks in advance
- **Flag high-risk batches** for additional in-process testing

The XGBoost model extracts 6 statistics (mean, std, slope, min, max, AUC) from each of 9 parameters over the first 48h, producing 54 features per batch.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Data simulation | Python / NumPy | Realistic CHO culture kinetics |
| Data storage | SQLite + SQLAlchemy | Structured batch + time-series tables |
| MSPC engine | scikit-learn PCA, SciPy | Industry-standard PCA/F-distribution UCLs |
| ML model | XGBoost + SHAP | High accuracy + explainability for process engineers |
| Dashboard | Plotly Dash + Bootstrap | Interactive, browser-based, no JS required |
| Config | YAML | Reproducible, environment-agnostic parameters |
| Testing | pytest | Unit tests for statistical correctness |

---

## Project Structure

```
bioprocess_dashboard/
├── data/
│   ├── raw/                    # Simulated or downloaded CSVs
│   ├── processed/              # Scaled, PCA-transformed datasets
│   └── sql/
│       └── bioprocess.db       # SQLite database (batches + measurements)
│
├── src/
│   ├── ingestion/
│   │   ├── simulate_data.py    # Bioreactor run data generator
│   │   └── load_to_sql.py      # CSV → SQLite loader
│   │
│   ├── monitoring/
│   │   ├── pca_engine.py       # PCA fit/transform, explained variance
│   │   ├── hotelling_t2.py     # T² statistic + UCL computation
│   │   └── spe.py              # Squared Prediction Error (Q statistic)
│   │
│   ├── modeling/
│   │   ├── feature_engineering.py   # Early-run feature extraction (0–48h)
│   │   ├── train_yield_model.py     # XGBoost training + CV
│   │   └── evaluate_model.py        # SHAP, RMSE, feature importance
│   │
│   └── dashboard/
│       ├── app.py              # Dash app entry point
│       ├── layout.py           # Page layout + tab structure
│       └── callbacks.py        # Interactive callback logic
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_mspc_analysis.ipynb  # PCA + T² walkthrough
│   └── 03_yield_modeling.ipynb # ML model development
│
├── tests/
│   ├── test_hotelling.py
│   ├── test_pca_engine.py
│   └── test_simulate_data.py
│
├── config/
│   └── params.yaml             # UCL alpha, PCA components, model hyperparams
│
├── docs/
│   └── methodology.md          # Statistical methods explained
│
├── requirements.txt
├── setup.py
└── README.md
```

---

---

## Quickstart

```bash
# 1. Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Simulate bioreactor data and load to SQL
python src/ingestion/simulate_data.py
python src/ingestion/load_to_sql.py

# 3. Fit MSPC models
python src/monitoring/pca_engine.py
python src/monitoring/hotelling_t2.py
python src/monitoring/spe.py

# 4. Engineer features and train yield model
python src/modeling/feature_engineering.py
python src/modeling/train_yield_model.py

# 5. Launch dashboard
python src/dashboard/app.py
# Open http://localhost:8050
```

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| **Batch Overview** | KPI cards (total / normal / marginal / OOC counts), sortable batch status table |
| **MSPC Charts** | Hotelling's T² control chart, SPE chart, PC1 vs PC2 score scatter |
| **Process Trends** | Time-series explorer — select any batch + any parameter |
| **Yield Predictor** | Select a batch → XGBoost predicts final titer from its 48h features |

---

## Dataset

Simulated to match typical fed-batch CHO cell culture:

- **80 batches** — 60 normal, 15 marginal, 5 out-of-control
- **14-day runs**, sampled every 4 hours (85 timepoints per batch)
- **11 process parameters**: pH, dissolved oxygen (DO%), temperature (°C), agitation (RPM), feed rate A, feed rate B, glucose (g/L), lactate (g/L), viable cell density (VCD, 10⁶ cells/mL), viability (%), titer (mg/L)

OOC batches are generated with amplified noise, pH drift, and suppressed titer — mimicking real failure modes like nutrient depletion or contamination events.

---

## Key References

- Jackson & Mudholkar (1979) — SPE UCL approximation
- Nomikos & MacGregor (1995) — Multivariate SPC for batch processes
- ICH Q10 — Pharmaceutical Quality System guideline