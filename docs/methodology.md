# Methodology: Bioprocess MSPC Dashboard

## 1. Data Generation

Synthetic CHO fed-batch culture data is generated to mirror real bioreactor outputs. Each batch spans 14 days with measurements every 4 hours (85 timepoints). Three batch populations are simulated:

| Class | Description | Perturbation |
|---|---|---|
| Normal | Within spec | Baseline noise |
| Marginal | Slight process drift | 2.5× noise, mild pH/titer drop |
| Out-of-Control (OOC) | Failed runs | 6× noise, severe drift, suppressed titer |

---

## 2. Multivariate Statistical Process Control (MSPC)

Standard univariate SPC (e.g. Shewhart charts) cannot capture correlations between process parameters (pH, DO, temperature, VCD). MSPC accounts for this covariance structure.

### 2.1 PCA — Dimension Reduction

Principal Component Analysis is applied to the mean-centered, unit-variance scaled measurement matrix **X** (n_timepoints × n_parameters).

The decomposition:

    X = T · P' + E

where:
- **T** = scores matrix (compressed representation of process state)
- **P** = loadings matrix (how each variable contributes to each PC)
- **E** = residual matrix (variation not captured by retained PCs)

We retain A=5 PCs (typically capturing >85% of variance).

### 2.2 Hotelling's T² — Detecting In-Space Deviations

The T² statistic is the multivariate equivalent of a Shewhart X̄ chart:

    T²_i = t_i · Λ⁻¹ · t_i'

where **Λ** = diagonal matrix of PC eigenvalues (variances).

High T² → the process has moved far from the reference operating region *within* the PCA model subspace.

**Phase-2 UCL** (for monitoring new batches):

    UCL = [A(n+1)(n-1) / n(n-A)] · F_{A, n-A, α}

### 2.3 SPE / Q Statistic — Detecting Out-of-Space Deviations

Squared Prediction Error measures how well the PCA model reconstructs the observation:

    SPE_i = ||e_i||² = ||X_i - X̂_i||²

High SPE → the batch is behaving in a structurally new way not present in the reference data (e.g. novel sensor fault, previously unseen failure mode).

The UCL uses the Jackson-Mudholkar (1979) approximation based on residual eigenvalues.

**Together T² and SPE provide complete MSPC coverage:**

| | T² Normal | T² High |
|---|---|---|
| **SPE Normal** | In control ✅ | Unusual direction, known structure |
| **SPE High** | New type of variation | Total process upset 🔴 |

---

## 3. Yield Prediction Model

### 3.1 Feature Engineering

From the first 48 hours of each batch run, 6 summary statistics are extracted per parameter:

| Statistic | Meaning |
|---|---|
| `mean` | Average process level |
| `std` | Variability / consistency |
| `slope` | Linear drift rate (OLS) |
| `min` | Minimum excursion |
| `max` | Maximum excursion |
| `auc` | Trapezoidal area (cumulative exposure) |

9 parameters × 6 statistics = **54 features per batch**.

### 3.2 XGBoost Regressor

- **Target**: Final batch titer (mg/L) at day 14
- **Validation**: 5-fold cross-validation on training set
- **Metrics**: RMSE, MAE, R²
- **Explainability**: SHAP TreeExplainer for feature attribution

Early features like mean VCD slope, pH AUC, and DO variability tend to be the strongest yield predictors.

---

## 4. References

- Jackson, J.E., Mudholkar, G.S. (1979). *Control procedures for residuals associated with principal component analysis*. Technometrics.
- Nomikos, P., MacGregor, J.F. (1995). *Multivariate SPC charts for monitoring batch processes*. Technometrics.
- Wold, S. et al. (1987). *Principal component analysis*. Chemometrics and Intelligent Laboratory Systems.
