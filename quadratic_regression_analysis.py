import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ======= CONFIG ==========
# =========================
tsv_path = "data.tsv"                  # <- change to your TSV path
x_col = "MEDIAN_LIST_PRICE"                  # predictor
y_col = "MEDIAN_SALE_PRICE"            # target

# Filters
filter_state = "CA"                    # Default to California as requested
filter_city  = None                    # e.g., "San Jose" to narrow further
date_start   = None                    # e.g., "2018-01-01"
date_end     = None                    # e.g., "2022-12-31"

# Modeling
use_log = True                         # True -> model on log(x), log(y)
split_by_time = True                   # <-- chronological split by default (uses PERIOD_BEGIN)
train_frac, val_frac, test_frac = 0.70, 0.15, 0.15
random_seed = 42                       # used only if time-split not available
min_rows_required = 5

# Output files
plot_fit_path = "quadratic_fit_splits.png"
plot_resid_vs_fitted_path = "residuals_vs_fitted_splits.png"
plot_resid_hist_path = "residual_hist_splits.png"
predictions_csv_path = "quadratic_predictions_splits.csv"

# =========================
# ======= LOAD ============
# =========================
df = pd.read_csv(tsv_path, sep="\t", low_memory=False, dtype=str)

# Parse numerics & date
for col in [x_col, y_col]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in file.")
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "PERIOD_BEGIN" in df.columns:
    df["PERIOD_BEGIN"] = pd.to_datetime(df["PERIOD_BEGIN"], errors="coerce")

# =========================
# ======= FILTERS =========
# =========================
if filter_state and "STATE_CODE" in df.columns:
    df = df[df["STATE_CODE"] == filter_state]

if filter_city and "CITY" in df.columns:
    df = df[df["CITY"] == filter_city]

if date_start and "PERIOD_BEGIN" in df.columns:
    df = df[df["PERIOD_BEGIN"] >= pd.to_datetime(date_start)]

if date_end and "PERIOD_BEGIN" in df.columns:
    df = df[df["PERIOD_BEGIN"] <= pd.to_datetime(date_end)]

# Keep only rows we can use
base_cols = [x_col, y_col] + (["PERIOD_BEGIN"] if "PERIOD_BEGIN" in df.columns else [])
work = df[base_cols].copy()
work = work.dropna(subset=[x_col, y_col])

if len(work) < min_rows_required:
    raise ValueError(f"Not enough rows after filtering: {len(work)} (need >= {min_rows_required}).")

# =========================
# ===== TRANSFORM =========
# =========================
if use_log:
    # Log requires positive values
    work = work[(work[x_col] > 0) & (work[y_col] > 0)]
    if len(work) < min_rows_required:
        raise ValueError("Not enough positive rows to log-transform.")
    work["x"] = np.log(work[x_col].astype(float))
    work["y"] = np.log(work[y_col].astype(float))
    x_label, y_label = f"log({x_col})", f"log({y_col})"
else:
    work["x"] = work[x_col].astype(float)
    work["y"] = work[y_col].astype(float)
    x_label, y_label = x_col, y_col

x = work["x"].to_numpy()
y = work["y"].to_numpy()
n_total = len(y)

# =========================
# ===== SPLITTING =========
# =========================
def design(x_arr: np.ndarray) -> np.ndarray:
    """Design matrix for quadratic regression: [1, x, x^2]."""
    return np.column_stack([np.ones_like(x_arr), x_arr, x_arr**2])

def medae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.median(np.abs(y_true - y_pred)))

def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error in %.
    Uses max(|y_true|, epsilon) in the denominator for stability with zeros.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def compute_metrics(y_true, y_pred, p=3, include_mape=True):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    n = len(y_true)
    RSS = float(np.sum(resid**2))
    TSS = float(np.sum((y_true - y_true.mean())**2))
    R2 = 1 - RSS/TSS if TSS > 0 else np.nan
    adjR2 = 1 - (RSS/(n - p)) / (TSS/(n - 1)) if n > p and TSS > 0 else np.nan
    RMSE = float(np.sqrt(RSS / n))
    MAE = float(np.mean(np.abs(resid)))
    MedAE = medae(y_true, y_pred)
    MAPEv = mape(y_true, y_pred) if include_mape else np.nan
    return dict(n=n, R2=R2, AdjR2=adjR2, RMSE=RMSE, MAE=MAE, MedAE=MedAE, MAPE=MAPEv)

def format_metrics_row(name, m, include_mape=True):
    base = (f"{name}: R2={m['R2']:.4f}, AdjR2={m['AdjR2']:.4f}, "
            f"RMSE={m['RMSE']:.3f}, MAE={m['MAE']:.3f}, MedAE={m['MedAE']:.3f}")
    if include_mape and not np.isnan(m['MAPE']):
        base += f", MAPE={m['MAPE']:.2f}%"
    return base

# Decide ordering for split (chronological by default if PERIOD_BEGIN exists)
idx_all = np.arange(n_total)
use_time_order = split_by_time and ("PERIOD_BEGIN" in work.columns)

if use_time_order:
    work = work.sort_values("PERIOD_BEGIN").reset_index(drop=True)
    x = work["x"].to_numpy()
    y = work["y"].to_numpy()
    idx_all = np.arange(len(y))

# Compute split sizes
assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "Fractions must sum to 1."
n_train = int(round(train_frac * n_total))
n_val   = int(round(val_frac   * n_total))
n_test  = n_total - n_train - n_val

# Ensure at least 1 sample in each split when possible
if n_total >= 3:
    if n_train < 1: n_train = 1
    if n_val < 1:   n_val   = 1
    if n_test < 1:  n_test  = 1
    # Recompute if the adjustments changed totals
    overflow = (n_train + n_val + n_test) - n_total
    if overflow > 0:
        # Reduce from the largest of the three as needed
        for name in ["n_train", "n_val", "n_test"]:
            if overflow == 0: break
            if eval(name) > 1:
                exec(f"{name} -= 1")
                overflow -= 1

# Build indices
if use_time_order:
    # Chronological: earliest -> train, then validation, latest -> test
    train_idx = idx_all[:n_train]
    val_idx   = idx_all[n_train:n_train + n_val]
    test_idx  = idx_all[n_train + n_val:]
else:
    # Random split (fallback when PERIOD_BEGIN is absent)
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(idx_all)
    train_idx = shuffled[:n_train]
    val_idx   = shuffled[n_train:n_train + n_val]
    test_idx  = shuffled[n_train + n_val:]

# ================================
# ===== FIT ON TRAIN ONLY ========
# ================================
X_train = design(x[train_idx]); y_train = y[train_idx]
X_val   = design(x[val_idx]);   y_val   = y[val_idx]
X_test  = design(x[test_idx]);  y_test  = y[test_idx]

beta, residuals_train, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)
a, b, c = beta

# Predictions per split
yhat_train = X_train @ beta
yhat_val   = X_val   @ beta
yhat_test  = X_test  @ beta

# Metrics on modeling scale
include_mape_on_this_scale = True  # If use_log, this is MAPE in log space (not price %).
m_train = compute_metrics(y_train, yhat_train, include_mape=include_mape_on_this_scale)
m_val   = compute_metrics(y_val,   yhat_val,   include_mape=include_mape_on_this_scale)
m_test  = compute_metrics(y_test,  yhat_test,  include_mape=include_mape_on_this_scale)

# Standard errors from TRAIN fit
resid_train = y_train - yhat_train
p = 3
sigma2 = np.sum(resid_train**2) / max(len(y_train) - p, 1)
cov_beta = np.linalg.inv(X_train.T @ X_train) * sigma2
se = np.sqrt(np.diag(cov_beta))

print("\n=== Quadratic fit (TRAIN) ===")
print(f"Using chronological split: {bool(use_time_order)}")
print(f"n_total={n_total}, n_train={m_train['n']}, n_val={m_val['n']}, n_test={m_test['n']}")
print(f"a={a:.6g}, b={b:.6g}, c={c:.6g}")
print(f"SE(a,b,c)={se}")
print(format_metrics_row("Train", m_train, include_mape_on_this_scale))
print(format_metrics_row("Valid", m_val,   include_mape_on_this_scale))
print(format_metrics_row(" Test", m_test,  include_mape_on_this_scale))
if use_log:
    print("(Note) MAPE above is on the log scale; see original-scale metrics below for % errors in prices.")

# ==================================
# ===== PLOTS (with split info) =====
# ==================================
xs = np.linspace(x.min(), x.max(), 300)
ys = a + b*xs + c*xs**2

plt.figure(figsize=(7.8,5.6))
plt.scatter(x[train_idx], y[train_idx], s=26, alpha=0.8, label="Train", color="#1f77b4")
plt.scatter(x[val_idx],   y[val_idx],   s=26, alpha=0.8, label="Validation", color="#ff7f0e")
plt.scatter(x[test_idx],  y[test_idx],  s=26, alpha=0.8, label="Test", color="#2ca02c")
plt.plot(xs, ys, color="crimson", lw=2.2, label="Quadratic fit (train only)")
plt.xlabel(x_label); plt.ylabel(y_label)
plt.title(f"Quadratic: {y_label} ~ f({x_label}) — Train/Val/Test (chronological)")
plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
plt.savefig(plot_fit_path, dpi=160)

plt.figure(figsize=(7.8,5.4))
plt.scatter(yhat_train, y_train - yhat_train, s=22, alpha=0.85, label="Train", color="#1f77b4")
plt.scatter(yhat_val,   y_val   - yhat_val,   s=22, alpha=0.85, label="Validation", color="#ff7f0e")
plt.scatter(yhat_test,  y_test  - yhat_test,  s=22, alpha=0.85, label="Test", color="#2ca02c")
plt.axhline(0, color="k", lw=1)
plt.xlabel("Fitted values"); plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (by split)")
plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
plt.savefig(plot_resid_vs_fitted_path, dpi=160)

plt.figure(figsize=(7.8,5.4))
plt.hist(y_train - yhat_train, bins="auto", alpha=0.7, label="Train", edgecolor="white")
plt.hist(y_val   - yhat_val,   bins="auto", alpha=0.7, label="Validation", edgecolor="white")
plt.hist(y_test  - yhat_test,  bins="auto", alpha=0.7, label="Test", edgecolor="white")
plt.xlabel("Residual"); plt.ylabel("Frequency")
plt.title("Residual Distribution (by split)")
plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
plt.savefig(plot_resid_hist_path, dpi=160)

# ===========================================
# ===== Export predictions & orig scale =====
# ===========================================
# Predictions for ALL rows with split labels
X_all = design(x)
yhat_all = X_all @ beta
resid_all = y - yhat_all

split = np.array([""] * n_total, dtype=object)
split[train_idx] = "train"; split[val_idx] = "validation"; split[test_idx] = "test"

out = work.assign(fitted=yhat_all, residual=resid_all, split=split)

# Back-transform and report original-scale metrics (if modeling on logs)
if use_log:
    # Duan smearing factor from train residuals
    smear = float(np.mean(np.exp(resid_train)))
    out["actual_orig"] = np.exp(out["y"])
    out["fitted_orig"] = np.exp(out["fitted"]) * smear

    def orig_metrics(df, which):
        w = df[df["split"] == which]
        y_true = w["actual_orig"].to_numpy()
        y_pred = w["fitted_orig"].to_numpy()
        resid_o = y_true - y_pred
        RMSE_o = float(np.sqrt(np.mean(resid_o**2)))
        MAE_o  = float(np.mean(np.abs(resid_o)))
        MedAE_o = medae(y_true, y_pred)
        MAPE_o  = mape(y_true, y_pred)
        return RMSE_o, MAE_o, MedAE_o, MAPE_o

    for sname in ["train", "validation", "test"]:
        RMSE_o, MAE_o, MedAE_o, MAPE_o = orig_metrics(out, sname)
        print(f"{sname.capitalize()} (original scale): "
              f"RMSE={RMSE_o:.2f}, MAE={MAE_o:.2f}, MedAE={MedAE_o:.2f}, MAPE={MAPE_o:.2f}%")

# Save CSV
out.to_csv(predictions_csv_path, index=False)

print("\nSaved outputs:")
print(f" - {plot_fit_path}")
print(f" - {plot_resid_vs_fitted_path}")
print(f" - {plot_resid_hist_path}")
print(f" - {predictions_csv_path}")
