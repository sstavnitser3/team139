#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time vs Sale Price - Linear vs Quadratic Trend with Seasonality, Piecewise Hinges,
Exogenous Lags, and Optional Rolling Backtest (with robust fixes)

What this script does
---------------------
- Filters a TSV to a geography (default: California; optional city) with optional date window.
- Aggregates to a single time series per date (median/mean) if multiple rows per date exist.
- Builds a time index in months from the earliest observation and adds seasonality sin/cos(month).
- Adds optional piecewise hinge features at user-specified knots (dates or month indices),
  optionally with quadratic hinges.
- Adds optional exogenous regressors with user-specified lags (e.g., INVENTORY_lag1, MoS_lag3).
- Fits two OLS models on TRAIN only with chronological splits:
    1) Linear:     y ~ a + b*t (+ d*sin + e*cos) + hinges + exog_lags
    2) Quadratic:  y ~ a + b*t + c*t^2 (+ d*sin + e*cos) + hinges (+ hinge^2) + exog_lags
- Supports log-target with Duan smearing back-transform (per model) for original-scale metrics/plots.
- Reports metrics (R2, AdjR2, RMSE, MAE, MedAE, MAPE) on modeling and original scales side-by-side,
  saves a metrics CSV.
- Saves overlay fit plot (linear vs quadratic), residual plots per model, combined residual histogram,
  and predictions CSV.
- Optional rolling backtest: fixed-size sliding training window with out-of-window test segments;
  outputs a CSV of rolling test metrics.

Run examples
------------
python time_trend_piecewise_exog.py --tsv data.tsv --state CA --use_log \
  --exog_cols INVENTORY MONTHS_OF_SUPPLY AVG_SALE_TO_LIST --exog_lags 1 3 6 \
  --knots 2020-04 2022-06 --piecewise_quadratic

python time_trend_piecewise_exog.py --tsv data.tsv --state CA --city "San Jose" \
  --aggregate median --use_log --rolling_window 60 --rolling_step 1

python time_trend_piecewise_exog.py --tsv data.tsv --state CA --use_log \
  --start 2015-01 --end 2023-12 --exog_cols INVENTORY --exog_lags 1 2 3
"""
import argparse
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def medae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.median(np.abs(y_true - y_pred)))

def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def compute_metrics(y_true, y_pred, p: int) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    n = max(1, len(y_true))
    RSS = float(np.sum(resid**2))
    TSS = float(np.sum((y_true - y_true.mean())**2))
    R2 = 1 - RSS/TSS if TSS > 0 else np.nan
    adjR2 = 1 - (RSS/(n - p)) / (TSS/(n - 1)) if n > p and TSS > 0 else np.nan
    RMSE = float(np.sqrt(RSS / n))
    MAE = float(np.mean(np.abs(resid)))
    MedAE = medae(y_true, y_pred)
    MAPEv = mape(y_true, y_pred)
    return dict(n=n, R2=R2, AdjR2=adjR2, RMSE=RMSE, MAE=MAE, MedAE=MedAE, MAPE=MAPEv)

def chronological_split(n: int, train_frac: float, val_frac: float, test_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    n_test = n - n_train - n_val
    if n_test < 1 and n >= 3:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

def build_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    per = out[date_col].dt.to_period('M').astype('int64')
    out['time_index'] = per - per.min()
    month = out[date_col].dt.month.astype(int)
    out['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    out['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    return out

def parse_knots_to_month_index(knots: List[str], work: pd.DataFrame, date_col: str) -> List[int]:
    if not knots:
        return []
    out = []
    base_ord = work[date_col].dt.to_period('M').astype('int64').min()
    for k in knots:
        ks = str(k)
        try:
            if len(ks) == 7 and ks[4] == '-':
                per = pd.Period(ks, freq='M')
            else:
                per = pd.Period(pd.to_datetime(ks), freq='M')
            out.append(int(per.ordinal - base_ord))
        except Exception:
            try:
                out.append(int(float(ks)))
            except Exception:
                warnings.warn(f"Could not parse knot '{k}', skipping.")
    tmin, tmax = int(work['time_index'].min()), int(work['time_index'].max())
    return [k for k in out if tmin < k < tmax]

def build_piecewise_columns(t: np.ndarray, knots: List[int], quadratic: bool = False) -> Dict[str, np.ndarray]:
    cols = {}
    if not knots:
        return cols
    for k in knots:
        hk = np.clip(t - k, 0, None)
        cols[f"hinge_{k}"] = hk
        if quadratic:
            cols[f"hinge2_{k}"] = hk**2
    return cols

def prune_constant_columns(X_train: np.ndarray, X_all: np.ndarray, col_names: List[str], eps: float = 1e-12):
    stds = X_train.std(axis=0)
    keep = stds > eps

    if 'Intercept' in col_names:
        keep[col_names.index('Intercept')] = True

    if not np.all(keep):
        dropped = [col_names[i] for i, k in enumerate(keep) if not k]
        print(f"[INFO] Dropping constant cols in TRAIN: {dropped}")
    return X_train[:, keep], X_all[:, keep], [n for n, k in zip(col_names, keep) if k], keep

def fit_ols(X_train: np.ndarray, y_train: np.ndarray):
    beta, _, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)
    yhat_tr = X_train @ beta
    resid_tr = y_train - yhat_tr
    p = X_train.shape[1]
    sigma2 = float(np.sum(resid_tr**2) / max(len(y_train) - p, 1))
    XtX = X_train.T @ X_train
    cov_beta = np.linalg.pinv(XtX) * sigma2
    se = np.sqrt(np.diag(cov_beta))
    return beta, sigma2, se, yhat_tr, rank, s

def run_rolling_backtest(dates, X_lin, X_quad, y, window, step, test_len, p_lin, p_quad):
    rows = []
    n = len(y)
    for end in range(window, n - test_len + 1, step):
        tr_start, tr_end = end - window, end
        te_start, te_end = end, end + test_len
        Xlin_tr, Xlin_te = X_lin[tr_start:tr_end],  X_lin[te_start:te_end]
        Xq_tr,   Xq_te   = X_quad[tr_start:tr_end], X_quad[te_start:te_end]
        y_tr,    y_te    = y[tr_start:tr_end],      y[te_start:te_end]
        blin, _, _, _, _, _ = fit_ols(Xlin_tr, y_tr)
        bquad, _, _, _, _, _ = fit_ols(Xq_tr, y_tr)
        yhat_te_lin = Xlin_te @ blin
        yhat_te_quad = Xq_te @ bquad
        m_lin = compute_metrics(y_te, yhat_te_lin, p_lin)
        m_quad = compute_metrics(y_te, yhat_te_quad, p_quad)
        rows.append({'window_end': pd.to_datetime(dates[tr_end-1]).date(),'test_end': pd.to_datetime(dates[te_end-1]).date(),'model': 'Linear','R2': m_lin['R2'],'RMSE': m_lin['RMSE'],'MAE': m_lin['MAE'],'MedAE': m_lin['MedAE'],'MAPE': m_lin['MAPE']})
        rows.append({'window_end': pd.to_datetime(dates[tr_end-1]).date(),'test_end': pd.to_datetime(dates[te_end-1]).date(),'model': 'Quadratic','R2': m_quad['R2'],'RMSE': m_quad['RMSE'],'MAE': m_quad['MAE'],'MedAE': m_quad['MedAE'],'MAPE': m_quad['MAPE']})
    return pd.DataFrame(rows)

# --------------
# Main routine
# --------------

def main():
    parser = argparse.ArgumentParser(description="Time trend with seasonality, piecewise hinges, exogenous lags; Linear vs Quadratic")
    parser.add_argument('--tsv', required=True, help='Path to TSV file')
    parser.add_argument('--date_col', default='PERIOD_BEGIN', help='Date column name')
    parser.add_argument('--y_col', default='MEDIAN_SALE_PRICE', help='Target column (sale price)')
    parser.add_argument('--state', default='CA', help='Filter by state code (e.g., CA)')
    parser.add_argument('--city', default=None, help='Optional city filter')
    parser.add_argument('--aggregate', choices=['none', 'median', 'mean'], default='median',
                        help='If multiple rows per date exist, aggregate to a single series')
    parser.add_argument('--use_log', action='store_true', help='Model log(y); uses Duan smearing for back-transform')
    parser.add_argument('--no_season', action='store_true', help='Disable seasonality (sin/cos of month)')
    parser.add_argument('--train', type=float, default=0.70, help='Train fraction')
    parser.add_argument('--val', type=float, default=0.15, help='Validation fraction')
    parser.add_argument('--test', type=float, default=0.15, help='Test fraction')
    parser.add_argument('--min_rows', type=int, default=12, help='Minimum rows required after filtering')
    parser.add_argument('--out_prefix', default='time_piecewise_exog', help='Output file prefix (plots/CSV)')

    # New: exogenous, lags, knots, rolling, date slice
    parser.add_argument('--exog_cols', nargs='*', default=[],
                        help='Optional exogenous columns to include (e.g., INVENTORY MONTHS_OF_SUPPLY AVG_SALE_TO_LIST)')
    parser.add_argument('--exog_lags', nargs='*', type=int, default=[0],
                        help='Lags in months to include for each exogenous column (e.g., 0 1 3 6)')
    parser.add_argument('--knots', nargs='*', default=[],
                        help='Piecewise knots as YYYY-MM or integer month indices (e.g., 2020-03 2022-06 or 60 90).')
    parser.add_argument('--piecewise_quadratic', action='store_true',
                        help='If set, add quadratic hinge terms ((t-k)^2_+) in addition to linear hinges.')
    parser.add_argument('--rolling_window', type=int, default=0,
                        help='If >0, run a rolling backtest using a trailing window of this many months for training.')
    parser.add_argument('--rolling_step', type=int, default=1,
                        help='Step in months between rolling windows (default 1).')
    parser.add_argument('--start', default=None, help='Optional inclusive start date YYYY-MM or YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='Optional inclusive end date YYYY-MM or YYYY-MM-DD')

    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.tsv, sep='\t', low_memory=False, dtype=str)

    if args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found.")
    if args.y_col not in df.columns:
        raise ValueError(f"Target column '{args.y_col}' not found.")

    # Parse target and date
    df[args.y_col] = pd.to_numeric(df[args.y_col], errors='coerce')
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors='coerce')

    # Geography filters
    if args.state and 'STATE_CODE' in df.columns:
        df = df[df['STATE_CODE'] == args.state]
    if args.city and 'CITY' in df.columns:
        df = df[df['CITY'] == args.city]

    # Parse exogenous columns
    for c in args.exog_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            warnings.warn(f"Exogenous column '{c}' not found; it will be ignored.")

    # Date window slices
    if args.start:
        df = df[df[args.date_col] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df[args.date_col] <= pd.to_datetime(args.end)]

    # Keep only necessary columns
    keep_cols = [args.date_col, args.y_col] + [c for c in args.exog_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=[args.date_col, args.y_col])

    if len(df) < args.min_rows:
        raise ValueError(f"Not enough rows after filtering: {len(df)} (need >= {args.min_rows}).")

    # Aggregate across entities per date if needed (Future-safe: use string names)
    df = df.sort_values(args.date_col)
    if args.aggregate != 'none':
        how = 'median' if args.aggregate == 'median' else 'mean'
        df = df.groupby(args.date_col, as_index=False)[keep_cols[1:]].agg(how)
        df = df.sort_values(args.date_col)

    # Build core time features
    work = build_time_features(df, args.date_col)

    # Build exogenous lags
    exog_lag_cols = []
    for col in args.exog_cols:
        if col not in work.columns:
            continue
        for L in sorted(set(args.exog_lags)):
            if L < 0:
                raise ValueError("--exog_lags must be >= 0")
            lag_col = f"{col}_lag{L}"
            if L == 0:
                work[lag_col] = work[col].astype(float)
            else:
                work[lag_col] = work[col].astype(float).shift(L)
            exog_lag_cols.append(lag_col)
    if exog_lag_cols:
        work = work.dropna(subset=exog_lag_cols)

    # Prepare y (optionally log)
    if args.use_log:
        work = work[work[args.y_col] > 0].copy()
        work['y'] = np.log(work[args.y_col].astype(float))
        y_label = f"log({args.y_col})"
    else:
        work['y'] = work[args.y_col].astype(float)
        y_label = args.y_col

    # IMPORTANT: ensure 0..n-1 indexing BEFORE building arrays / splitting / exporting
    work = work.sort_values(args.date_col).reset_index(drop=True)

    # Arrays and seasonality
    t = work['time_index'].to_numpy()
    y = work['y'].to_numpy()
    add_season = not args.no_season
    sinv = work['month_sin'].to_numpy() if add_season else None
    cosv = work['month_cos'].to_numpy() if add_season else None

    # Piecewise hinges
    knots_idx = parse_knots_to_month_index(args.knots, work, args.date_col)
    hinge_cols = build_piecewise_columns(t, knots_idx, quadratic=args.piecewise_quadratic)

    # --- Build feature columns & names in the same order (for pruning & SE labeling) ---
    feat_cols_lin_names  = ['Intercept', 't']
    feat_cols_quad_names = ['Intercept', 't', 't2']

    cols_lin  = [np.ones_like(t), t.astype(float)]
    cols_quad = [np.ones_like(t), t.astype(float), (t.astype(float))**2]

    if add_season:
        cols_lin  += [sinv.astype(float), cosv.astype(float)]
        cols_quad += [sinv.astype(float), cosv.astype(float)]
        feat_cols_lin_names  += ['sin', 'cos']
        feat_cols_quad_names += ['sin', 'cos']

    # hinges (same order as created)
    for name in hinge_cols:
        h = hinge_cols[name].astype(float)
        cols_lin.append(h);  cols_quad.append(h)
        feat_cols_lin_names.append(name); feat_cols_quad_names.append(name)

    # exogenous lags
    for c in exog_lag_cols:
        v = work[c].to_numpy().astype(float)
        cols_lin.append(v); cols_quad.append(v)
        feat_cols_lin_names.append(c); feat_cols_quad_names.append(c)

    X_lin  = np.column_stack(cols_lin)
    X_quad = np.column_stack(cols_quad)

    # Chronological split
    n = len(y)
    p_lin  = X_lin.shape[1]
    p_quad = X_quad.shape[1]
    if n < max(p_quad + 3, 10):
        warnings.warn("Very few rows relative to parameters; estimates may be unstable.")
    train_idx, val_idx, test_idx = chronological_split(n, args.train, args.val, args.test)

    # Train/Val/Test partitions
    Xlin_tr, Xlin_va, Xlin_te = X_lin[train_idx],  X_lin[val_idx],  X_lin[test_idx]
    Xq_tr,   Xq_va,   Xq_te   = X_quad[train_idx], X_quad[val_idx], X_quad[test_idx]
    y_tr,    y_va,    y_te    = y[train_idx],      y[val_idx],      y[test_idx]

    # Prune constant columns in TRAIN (prevents rank deficiency from all-zero hinges, etc.)
    Xlin_tr, X_lin,  feat_cols_lin_names,  keep_lin  = prune_constant_columns(Xlin_tr, X_lin,  feat_cols_lin_names)
    Xq_tr,   X_quad, feat_cols_quad_names, keep_quad = prune_constant_columns(Xq_tr,   X_quad, feat_cols_quad_names)
    # Re-slice pruned X for val/test after pruning the full matrices
    Xlin_va, Xlin_te = X_lin[val_idx],  X_lin[test_idx]
    Xq_va,   Xq_te   = X_quad[val_idx], X_quad[test_idx]
    # Update p after pruning
    p_lin  = X_lin.shape[1]
    p_quad = X_quad.shape[1]

    # Fit models on TRAIN
    blin, sig2_lin, se_lin, yhat_tr_lin, _, _ = fit_ols(Xlin_tr, y_tr)
    bquad, sig2_quad, se_quad, yhat_tr_quad, _, _ = fit_ols(Xq_tr, y_tr)

    # Predictions on splits (model scale)
    yhat_va_lin = Xlin_va @ blin
    yhat_te_lin = Xlin_te @ blin
    yhat_va_quad = Xq_va @ bquad
    yhat_te_quad = Xq_te @ bquad

    # Metrics on model scale
    m_lin_train  = compute_metrics(y_tr, yhat_tr_lin, p_lin)
    m_lin_val    = compute_metrics(y_va, yhat_va_lin, p_lin)
    m_lin_test   = compute_metrics(y_te, yhat_te_lin, p_lin)

    m_quad_train = compute_metrics(y_tr, yhat_tr_quad, p_quad)
    m_quad_val   = compute_metrics(y_va, yhat_va_quad, p_quad)
    m_quad_test  = compute_metrics(y_te, yhat_te_quad, p_quad)

    # Predictions for ALL rows for plotting/export (use pruned full matrices)
    yhat_all_lin  = X_lin  @ blin
    yhat_all_quad = X_quad @ bquad

    # Collect metrics rows
    metrics_rows = []
    def add_metrics(model_name: str, split_name: str, m: Dict[str, float], scale: str):
        metrics_rows.append(dict(Model=model_name, Split=split_name, Scale=scale,
                                 R2=m['R2'], AdjR2=m['AdjR2'], RMSE=m['RMSE'], MAE=m['MAE'], MedAE=m['MedAE'], MAPE=m['MAPE']))

    for name, tr, va, te in [
        ('Linear', m_lin_train, m_lin_val, m_lin_test),
        ('Quadratic', m_quad_train, m_quad_val, m_quad_test)
    ]:
        add_metrics(name, 'train', tr, 'model')
        add_metrics(name, 'validation', va, 'model')
        add_metrics(name, 'test', te, 'model')

    # Original-scale metrics (if log)
    if args.use_log:
        smear_lin  = float(np.mean(np.exp(y_tr - yhat_tr_lin)))
        smear_quad = float(np.mean(np.exp(y_tr - yhat_tr_quad)))
        y_tr_o = np.exp(y_tr); y_va_o = np.exp(y_va); y_te_o = np.exp(y_te)
        yhat_tr_lin_o = np.exp(yhat_tr_lin) * smear_lin
        yhat_va_lin_o = np.exp(yhat_va_lin) * smear_lin
        yhat_te_lin_o = np.exp(yhat_te_lin) * smear_lin
        yhat_tr_quad_o = np.exp(yhat_tr_quad) * smear_quad
        yhat_va_quad_o = np.exp(yhat_va_quad) * smear_quad
        yhat_te_quad_o = np.exp(yhat_te_quad) * smear_quad

        m_lin_train_o  = compute_metrics(y_tr_o, yhat_tr_lin_o, p_lin)
        m_lin_val_o    = compute_metrics(y_va_o, yhat_va_lin_o, p_lin)
        m_lin_test_o   = compute_metrics(y_te_o, yhat_te_lin_o, p_lin)

        m_quad_train_o = compute_metrics(y_tr_o, yhat_tr_quad_o, p_quad)
        m_quad_val_o   = compute_metrics(y_va_o, yhat_va_quad_o, p_quad)
        m_quad_test_o  = compute_metrics(y_te_o, yhat_te_quad_o, p_quad)

        for name, tr, va, te in [
            ('Linear', m_lin_train_o, m_lin_val_o, m_lin_test_o),
            ('Quadratic', m_quad_train_o, m_quad_val_o, m_quad_test_o)
        ]:
            add_metrics(name, 'train', tr, 'original')
            add_metrics(name, 'validation', va, 'original')
            add_metrics(name, 'test', te, 'original')

    # ---- Console summary ----
    def coef_summary(beta: np.ndarray, se: np.ndarray, names: List[str]) -> str:
        parts = []
        for nm, v, s in zip(names, beta, se):
            parts.append(f"{nm}={v:.6g} (SE={s:.6g})")
        return ", ".join(parts)

    print("\n=== LINEAR time trend (TRAIN fit) ===")
    print(coef_summary(blin, se_lin, feat_cols_lin_names))
    print(f"Train: R2={m_lin_train['R2']:.4f}, AdjR2={m_lin_train['AdjR2']:.4f}, RMSE={m_lin_train['RMSE']:.3f}, MAE={m_lin_train['MAE']:.3f}, MedAE={m_lin_train['MedAE']:.3f}, MAPE={m_lin_train['MAPE']:.2f}%")
    print(f"Valid: R2={m_lin_val['R2']:.4f}, AdjR2={m_lin_val['AdjR2']:.4f}, RMSE={m_lin_val['RMSE']:.3f}, MAE={m_lin_val['MAE']:.3f}, MedAE={m_lin_val['MedAE']:.3f}, MAPE={m_lin_val['MAPE']:.2f}%")
    print(f" Test: R2={m_lin_test['R2']:.4f}, AdjR2={m_lin_test['AdjR2']:.4f}, RMSE={m_lin_test['RMSE']:.3f}, MAE={m_lin_test['MAE']:.3f}, MedAE={m_lin_test['MedAE']:.3f}, MAPE={m_lin_test['MAPE']:.2f}%")

    print("\n=== QUADRATIC time trend (TRAIN fit) ===")
    print(coef_summary(bquad, se_quad, feat_cols_quad_names))
    print(f"Train: R2={m_quad_train['R2']:.4f}, AdjR2={m_quad_train['AdjR2']:.4f}, RMSE={m_quad_train['RMSE']:.3f}, MAE={m_quad_train['MAE']:.3f}, MedAE={m_quad_train['MedAE']:.3f}, MAPE={m_quad_train['MAPE']:.2f}%")
    print(f"Valid: R2={m_quad_val['R2']:.4f}, AdjR2={m_quad_val['AdjR2']:.4f}, RMSE={m_quad_val['RMSE']:.3f}, MAE={m_quad_val['MAE']:.3f}, MedAE={m_quad_val['MedAE']:.3f}, MAPE={m_quad_val['MAPE']:.2f}%")
    print(f" Test: R2={m_quad_test['R2']:.4f}, AdjR2={m_quad_test['AdjR2']:.4f}, RMSE={m_quad_test['RMSE']:.3f}, MAE={m_quad_test['MAE']:.3f}, MedAE={m_quad_test['MedAE']:.3f}, MAPE={m_quad_test['MAPE']:.2f}%")
    if args.use_log:
        print("(Note) MAPE above is on the model (log) scale; original-scale metrics (with Duan smearing) are saved to CSV.")

    # --------------
    # Visualization (overlay fits)
    # --------------
    dates = work[args.date_col].to_numpy()
    if args.use_log:
        smear_lin  = float(np.mean(np.exp(y_tr - yhat_tr_lin)))
        smear_quad = float(np.mean(np.exp(y_tr - yhat_tr_quad)))
        line_lin  = np.exp(yhat_all_lin)  * smear_lin
        line_quad = np.exp(yhat_all_quad) * smear_quad
        ydata = np.exp(y)
        ylabel = args.y_col
    else:
        line_lin  = yhat_all_lin
        line_quad = yhat_all_quad
        ydata = y
        ylabel = y_label

    order = np.argsort(dates)

    plt.figure(figsize=(10.2, 6.0))
    plt.scatter(dates[train_idx], ydata[train_idx], s=22, alpha=0.85, label='Train', color='#1f77b4')
    plt.scatter(dates[val_idx],   ydata[val_idx],   s=22, alpha=0.85, label='Validation', color='#ff7f0e')
    plt.scatter(dates[test_idx],  ydata[test_idx],  s=22, alpha=0.85, label='Test', color='#2ca02c')
    plt.plot(dates[order], line_lin[order],  color='black',   lw=2.0, label='Linear fit')
    plt.plot(dates[order], line_quad[order], color='crimson', lw=2.0, label='Quadratic fit')
    plt.xlabel('Date'); plt.ylabel(ylabel)
    plt.title('Sale Price vs Time - Linear vs Quadratic Trend (seasonality, hinges, exogenous)')
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_fit_linear_vs_quadratic.png", dpi=160)

    # Residuals vs fitted for Linear (model scale)
    plt.figure(figsize=(8.6, 5.2))
    plt.scatter(yhat_tr_lin, y_tr - yhat_tr_lin, s=20, alpha=0.85, label='Train', color='#1f77b4')
    plt.scatter(yhat_va_lin, y_va - yhat_va_lin, s=20, alpha=0.85, label='Validation', color='#ff7f0e')
    plt.scatter(yhat_te_lin, y_te - yhat_te_lin, s=20, alpha=0.85, label='Test', color='#2ca02c')
    plt.axhline(0, color='k', lw=1)
    plt.xlabel('Fitted (model scale)'); plt.ylabel('Residual (model scale)')
    plt.title('Residuals vs Fitted - LINEAR')
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_residuals_linear.png", dpi=160)

    # Residuals vs fitted for Quadratic (model scale)
    plt.figure(figsize=(8.6, 5.2))
    plt.scatter(yhat_tr_quad, y_tr - yhat_tr_quad, s=20, alpha=0.85, label='Train', color='#1f77b4')
    plt.scatter(yhat_va_quad, y_va - yhat_va_quad, s=20, alpha=0.85, label='Validation', color='#ff7f0e')
    plt.scatter(yhat_te_quad, y_te - yhat_te_quad, s=20, alpha=0.85, label='Test', color='#2ca02c')
    plt.axhline(0, color='k', lw=1)
    plt.xlabel('Fitted (model scale)'); plt.ylabel('Residual (model scale)')
    plt.title('Residuals vs Fitted - QUADRATIC')
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_residuals_quadratic.png", dpi=160)

    # Residual histogram (model scale) - combined
    plt.figure(figsize=(8.6, 5.2))
    plt.hist(y - yhat_all_lin, bins='auto', alpha=0.6, edgecolor='white', label='Linear')
    plt.hist(y - yhat_all_quad, bins='auto', alpha=0.6, edgecolor='white', label='Quadratic')
    plt.xlabel('Residual (model scale)'); plt.ylabel('Frequency')
    plt.title('Residual Distribution - Linear vs Quadratic (model scale)')
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_residual_hist_both.png", dpi=160)

    # --------------
    # Export CSVs
    # --------------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = f"{args.out_prefix}_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)

    out = work[[args.date_col, args.y_col]].copy()
    out['time_index'] = work['time_index']
    out['split'] = ''
    out.loc[train_idx, 'split'] = 'train'
    out.loc[val_idx,   'split'] = 'validation'
    out.loc[test_idx,  'split'] = 'test'

    out['fitted_linear_model']   = yhat_all_lin
    out['residual_linear_model'] = y - yhat_all_lin
    out['fitted_quadr_model']    = yhat_all_quad
    out['residual_quadr_model']  = y - yhat_all_quad

    if args.use_log:
        out['actual_orig']          = np.exp(y)
        out['fitted_linear_orig']   = line_lin
        out['residual_linear_orig'] = out['actual_orig'] - out['fitted_linear_orig']
        out['fitted_quadr_orig']    = line_quad
        out['residual_quadr_orig']  = out['actual_orig'] - out['fitted_quadr_orig']
        out['smearing_linear']      = float(np.mean(np.exp(y_tr - yhat_tr_lin)))
        out['smearing_quadr']       = float(np.mean(np.exp(y_tr - yhat_tr_quad)))

    preds_path = f"{args.out_prefix}_predictions.csv"
    out.to_csv(preds_path, index=False)

    print("\nSaved outputs:")
    print(f" - Overlay fit: {args.out_prefix}_fit_linear_vs_quadratic.png")
    print(f" - Residuals (linear): {args.out_prefix}_residuals_linear.png")
    print(f" - Residuals (quadratic): {args.out_prefix}_residuals_quadratic.png")
    print(f" - Residual hist (both): {args.out_prefix}_residual_hist_both.png")
    print(f" - Metrics summary: {metrics_path}")
    print(f" - Predictions: {preds_path}")

    # Rolling backtest
    if args.rolling_window > 0:
        test_len = max(1, int(round(args.test * len(y))))
        rb = run_rolling_backtest(
            dates=work[args.date_col].to_numpy(),
            X_lin=X_lin, X_quad=X_quad, y=y,
            window=args.rolling_window, step=args.rolling_step,
            test_len=test_len, p_lin=p_lin, p_quad=p_quad
        )
        rb_path = f"{args.out_prefix}_rolling_backtest.csv"
        rb.to_csv(rb_path, index=False)
        print(f" - Rolling backtest: {rb_path}")


if __name__ == '__main__':
    main()