from __future__ import annotations

import os
import json
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sqlalchemy import select, and_, create_engine
from sqlalchemy.orm import Session

from ml_engine.ingestion.models import Observation


def persistence_baseline(df: pd.DataFrame) -> pd.Series:
    """Predict future precipitation as the last observed value (persistence).

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame with a `precip_mm` column indexed by UTC datetime.

    Returns
    -------
    pandas.Series
        Predicted precipitation series aligned to the input index (uses only past/current).
    """
    return df["precip_mm"].astype(float)


def moving_average_baseline(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Predict future precipitation as the rolling mean of the last `window` observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame with a `precip_mm` column indexed by UTC datetime.
    window : int
        Number of past hours to average (default 3).

    Returns
    -------
    pandas.Series
        Predicted precipitation rolling average aligned to the input index.
    """
    return df["precip_mm"].rolling(window, min_periods=window).mean().astype(float)


def compute_labels(df: pd.DataFrame, horizon_steps: int) -> Tuple[pd.Series, pd.Series]:
    """Compute supervised labels shifted forward by `horizon_steps` hours.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame containing `precip_mm`.
    horizon_steps : int
        Number of hourly steps into the future to predict.

    Returns
    -------
    y_reg : pandas.Series
        Future precipitation amount in mm aligned at current time.
    y_clf : pandas.Series
        Binary label indicating rain (> 0.0) in the future horizon.
    """
    y_reg = df["precip_mm"].shift(-horizon_steps).astype(float)
    y_clf = (y_reg > 0.0).astype(int)
    return y_reg, y_clf


def compute_labels_by_location(df: pd.DataFrame, horizon_steps: int) -> Tuple[pd.Series, pd.Series]:
    """Compute labels per-location to avoid leakage across interleaved locations."""
    if "location_id" not in df.columns:
        return compute_labels(df, horizon_steps)

    y_reg_parts = []
    y_clf_parts = []
    for _, grp in df.groupby("location_id"):
        y_reg_loc = grp["precip_mm"].shift(-horizon_steps).astype(float)
        y_clf_loc = (y_reg_loc > 0.0).astype(int)
        y_reg_parts.append(y_reg_loc)
        y_clf_parts.append(y_clf_loc)
    y_reg = pd.concat(y_reg_parts).sort_index()
    y_clf = pd.concat(y_clf_parts).sort_index()
    return y_reg, y_clf


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Compute Brier score for binary outcomes.

    Parameters
    ----------
    y_true : pandas.Series
        Binary ground truth labels (0/1).
    y_prob : pandas.Series
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Mean squared error between probabilities and binary outcomes.
    """
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def evaluate_metrics(
    df: pd.DataFrame,
    horizon_steps: int,
    y_pred_clf_proba: Optional[pd.Series] = None,
    y_pred_reg: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Evaluate baselines and optional model predictions with classification and regression metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with `precip_mm` column and UTC datetime index.
    horizon_steps : int
        Number of hourly steps to predict into the future.
    y_pred_clf_proba : Optional[pandas.Series]
        Model-predicted probabilities of rain in the future horizon, aligned to `df.index`.
    y_pred_reg : Optional[pandas.Series]
        Model-predicted precipitation amount for the future horizon, aligned to `df.index`.

    Returns
    -------
    Dict[str, float]
        Metrics dictionary with baseline and optional model metrics.
    """
    # Compute labels
    y_reg, y_clf = compute_labels_by_location(df, horizon_steps)

    # Align and drop rows with NaNs introduced by shifting/rolling
    aligned = pd.DataFrame({
        "y_reg": y_reg,
        "y_clf": y_clf,
        "persist_reg": persistence_baseline(df),
        "ma_reg": moving_average_baseline(df),
    }).dropna(subset=["y_reg", "y_clf"])  # drop rows without future label or insufficient MA history

    # Baseline classification probabilities: 1 if baseline reg > 0 else 0
    aligned["persist_clf_prob"] = (aligned["persist_reg"] > 0.0).astype(float)
    aligned["ma_clf_prob"] = (aligned["ma_reg"] > 0.0).astype(float)

    metrics: Dict[str, float] = {}

    # Classification metrics
    for name in ["persist", "ma"]:
        prob = aligned[f"{name}_clf_prob"]
        mask = aligned[f"{name}_reg"].notna()  # ensure no NaNs in predictions
        y_clf_masked = aligned["y_clf"][mask]
        prob_masked = prob[mask]
        try:
            metrics[f"{name}_roc_auc"] = float(roc_auc_score(y_clf_masked, prob_masked))
        except Exception:
            metrics[f"{name}_roc_auc"] = float("nan")
        metrics[f"{name}_brier"] = brier_score(y_clf_masked, prob_masked)

    if y_pred_clf_proba is not None:
        proba = pd.Series(y_pred_clf_proba, index=df.index).loc[aligned.index]
        try:
            metrics["model_roc_auc"] = float(roc_auc_score(aligned["y_clf"], proba))
        except Exception:
            metrics["model_roc_auc"] = float("nan")
        metrics["model_brier"] = brier_score(aligned["y_clf"], proba)

    # Regression metrics
    for name in ["persist", "ma"]:
        yhat = aligned[f"{name}_reg"]
        mask = yhat.notna()
        metrics[f"{name}_mae"] = float(mean_absolute_error(aligned["y_reg"][mask], yhat[mask]))
        metrics[f"{name}_rmse"] = float(mean_squared_error(aligned["y_reg"][mask], yhat[mask])) ** 0.5

    if y_pred_reg is not None:
        yhat = pd.Series(y_pred_reg, index=df.index).loc[aligned.index]
        metrics["model_mae"] = float(mean_absolute_error(aligned["y_reg"], yhat))
        metrics["model_rmse"] = float(mean_squared_error(aligned["y_reg"], yhat)) ** 0.5

    return metrics


def save_report(metrics: Dict[str, float], md_path: str, json_path: str) -> None:
    """Save a markdown report and JSON metrics file.

    Parameters
    ----------
    metrics : Dict[str, float]
        Metrics produced by `evaluate`.
    md_path : str
        Path to write markdown report (e.g., `docs/ml_report.md`).
    json_path : str
        Path to write JSON metrics (e.g., `ml_engine/artifacts/metrics.json`).
    """
    # Markdown
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    lines = [
        "# ML Evaluation Report",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in sorted(metrics.items()):
        lines.append(f"| {k} | {v:.6f} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # JSON
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# Plotting functions (matplotlib only)

def plot_precip_predictions(
    df: pd.DataFrame,
    horizon_steps: int,
    y_pred_reg_dict: Optional[Dict[str, pd.Series]] = None,
) -> plt.Figure:
    """Plot actual future precipitation vs baseline/model predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame with `precip_mm`.
    horizon_steps : int
        Future horizon steps.
    y_pred_reg_dict : Optional[Dict[str, pandas.Series]]
        Mapping from label to prediction series aligned to `df.index`.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    y_reg, _ = compute_labels(df, horizon_steps)
    fig, ax = plt.subplots(figsize=(10, 4))
    y_reg.plot(ax=ax, label="future_precip_mm")
    persist = persistence_baseline(df)
    persist.loc[y_reg.index].plot(ax=ax, label="persistence")
    ma = moving_average_baseline(df)
    ma.loc[y_reg.index].plot(ax=ax, label="moving_avg_3h")
    if y_pred_reg_dict:
        for name, series in y_pred_reg_dict.items():
            pd.Series(series, index=df.index).loc[y_reg.index].plot(ax=ax, label=name)
    ax.set_title("Precipitation Predictions vs Actual")
    ax.set_ylabel("mm")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(
    df: pd.DataFrame,
    horizon_steps: int,
    y_pred_reg_dict: Optional[Dict[str, pd.Series]] = None,
) -> plt.Figure:
    """Plot residuals (prediction - actual) for baselines/models.

    Returns a figure with lines of residuals over time.
    """
    y_reg, _ = compute_labels(df, horizon_steps)
    fig, ax = plt.subplots(figsize=(10, 4))
    base_persist = (persistence_baseline(df).loc[y_reg.index] - y_reg)
    base_ma = (moving_average_baseline(df).loc[y_reg.index] - y_reg)
    base_persist.plot(ax=ax, label="persist_resid")
    base_ma.plot(ax=ax, label="ma_resid")
    if y_pred_reg_dict:
        for name, series in y_pred_reg_dict.items():
            resid = pd.Series(series, index=df.index).loc[y_reg.index] - y_reg
            resid.plot(ax=ax, label=f"{name}_resid")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Residuals")
    ax.set_ylabel("mm")
    ax.legend()
    fig.tight_layout()
    return fig


def evaluate_and_save(
    df: pd.DataFrame,
    horizon_steps: int,
    md_path: str = "docs/ml_report.md",
    json_path: str = "ml_engine/artifacts/metrics.json",
    y_pred_clf_proba: Optional[pd.Series] = None,
    y_pred_reg: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Convenience function to compute metrics and save report files."""
    metrics = evaluate_metrics(df, horizon_steps, y_pred_clf_proba=y_pred_clf_proba, y_pred_reg=y_pred_reg)
    save_report(metrics, md_path=md_path, json_path=json_path)
    return metrics


def _ceil_hours(minutes: int) -> int:
    return max(1, int(np.ceil(minutes / 60)))


def _load_observations(db_url: str, location_ids: Iterable[int], start_dt, end_dt) -> pd.DataFrame:
    engine = create_engine(db_url, future=True)
    with Session(engine) as session:
        cond = and_(
            Observation.location_id.in_(list(location_ids)),
            Observation.ts_utc >= start_dt,
            Observation.ts_utc < end_dt,
        )
        rows = session.execute(select(Observation).where(cond).order_by(Observation.ts_utc)).scalars().all()

    if not rows:
        raise ValueError("No observations found for given parameters.")

    idx = pd.to_datetime([r.ts_utc for r in rows], utc=True)
    df = pd.DataFrame(
        {
            "temp_c": [r.temp_c for r in rows],
            "humidity": [r.humidity for r in rows],
            "pressure_hpa": [r.pressure_hpa for r in rows],
            "wind_ms": [r.wind_ms for r in rows],
            "precip_mm": [r.precip_mm for r in rows],
            "wind_gust_ms": [r.wind_gust_ms for r in rows],
            "cloud_cover": [r.cloud_cover for r in rows],
            "location_id": [r.location_id for r in rows],
        },
        index=idx,
    ).sort_index()
    return df


def _parse_args():
    import argparse, datetime as dt
    p = argparse.ArgumentParser(description="Evaluate baselines and optional model metrics")
    p.add_argument("--db-url", required=True, help="SQLAlchemy database URL")
    p.add_argument("--location-ids", required=True, help="Comma-separated location IDs")
    p.add_argument("--start", required=True, help="Start datetime ISO (UTC)")
    p.add_argument("--end", required=True, help="End datetime ISO (UTC)")
    p.add_argument("--horizon", default="60", help="Future horizon in minutes (default 60)")
    p.add_argument("--md-path", default="docs/ml_report.md", help="Path for markdown report")
    p.add_argument("--json-path", default="ml_engine/artifacts/metrics.json", help="Path for JSON metrics")
    args = p.parse_args()

    location_ids = [int(x) for x in args.location_ids.split(",") if x.strip()]
    horizon_min = int(args.horizon)

    def to_utc(s: str):
        d = pd.to_datetime(s, utc=True)
        return d.to_pydatetime()

    return (
        args.db_url,
        location_ids,
        to_utc(args.start),
        to_utc(args.end),
        horizon_min,
        args.md_path,
        args.json_path,
    )


def main():
    db_url, loc_ids, start_dt, end_dt, horizon_min, md_path, json_path = _parse_args()
    df = _load_observations(db_url, loc_ids, start_dt, end_dt)
    # Merge all locations; evaluate on the combined series by simple concat
    # Labels are computed on the combined series which is acceptable for baseline evaluation
    steps = _ceil_hours(horizon_min)
    metrics = evaluate_and_save(df, steps, md_path=md_path, json_path=json_path)
    print(json.dumps(metrics, indent=2))


# Legacy helper to keep smoke tests working (expects callable named `evaluate`)
def evaluate(model: Dict[str, float], X: list, y: list) -> Dict[str, float]:
    """Compute MAE for a constant-bias model over given targets.

    Parameters
    ----------
    model : Dict[str, float]
        Model parameters containing a `bias` term.
    X : list
        Feature vectors (unused for bias-only model).
    y : list
        Targets.

    Returns
    -------
    Dict[str, float]
        Mapping containing `mae`.
    """
    bias = float(model.get("bias", 0.0))
    preds = np.full(len(y), bias, dtype=float)
    mae = float(mean_absolute_error(y, preds)) if len(y) else 0.0
    rmse = float(mean_squared_error(y, preds)) ** 0.5 if len(y) else 0.0
    return {"mae": mae, "rmse": rmse}
