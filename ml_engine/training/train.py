from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
import logging
from typing import Iterable, List, Tuple, Dict

import numpy as np
import pandas as pd
from sqlalchemy import select, and_, create_engine
from sqlalchemy.orm import Session

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
import joblib

from ml_engine.features.build_features import build_features
from ml_engine.ingestion.models import Observation


@dataclass
class TrainConfig:
    db_url: str
    location_ids: List[int]
    start_dt: dt.datetime
    end_dt: dt.datetime
    horizons_min: List[int]
    valid_fraction: float = 0.2
    artifacts_dir: str = os.path.join("ml_engine", "artifacts")


def _ceil_hours(minutes: int) -> int:
    """Map minute horizons to whole-hour steps (generic API).

    Examples
    --------
    15 -> 1, 30 -> 1, 60 -> 1, 120 -> 2
    """
    return max(1, math.ceil(minutes / 60))


def _load_observations(db_url: str, location_ids: Iterable[int], start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    """Load observations for given locations and date range.

    Returns a DataFrame indexed by UTC datetimes with required columns.
    """
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


def _build_supervised(df: pd.DataFrame, horizon_min: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create feature matrix and labels per-location with future shift.

    Ensures no leakage across locations by computing features and labels
    independently for each `location_id`, then concatenating.

    Returns
    -------
    X : DataFrame
        Feature matrix aligned so each row uses only past data.
    y_clf : Series
        Binary label indicating rain in next horizon (precip_mm > 0).
    y_reg : Series
        Continuous label of precipitation amount in next horizon (mm).
    """
    horizon_steps = _ceil_hours(horizon_min)

    parts: List[pd.DataFrame] = []
    feature_cols: List[str] = []
    for loc_id, grp in df.groupby("location_id"):
        base = grp.drop(columns=[c for c in ["location_id"] if c in grp.columns])
        feats = build_features(base)
        if not feature_cols:
            feature_cols = list(feats.columns)
        # Future labels per location
        y_future = base["precip_mm"].shift(-horizon_steps)
        aligned = pd.concat(
            [feats, (y_future > 0.0).astype(int).rename("y_clf"), y_future.astype(float).rename("y_reg")],
            axis=1,
        )
        aligned["location_id"] = loc_id
        parts.append(aligned)

    all_aligned = pd.concat(parts).sort_index()

    # Make the index unique across pooled locations.
    # When pooling multiple locations, timestamps repeat across locations; using a MultiIndex
    # avoids ambiguous alignment and boolean-mask indexing errors.
    ts_idx = pd.to_datetime(all_aligned.index, utc=True)
    loc_idx = all_aligned["location_id"].astype(int).to_numpy()
    all_aligned.index = pd.MultiIndex.from_arrays([ts_idx, loc_idx], names=["ts_utc", "location_id"])
    all_aligned = all_aligned.drop(columns=["location_id"])  # now represented in the index

    all_aligned = all_aligned.dropna(subset=["y_clf", "y_reg"]).copy()
    X = all_aligned[feature_cols].fillna(0.0)
    y_clf = all_aligned["y_clf"].astype(int)
    y_reg = all_aligned["y_reg"].astype(float)
    return X, y_clf, y_reg


def _time_split(X: pd.DataFrame, y: pd.Series, valid_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split by timestamp (not random), to avoid leakage."""
    n = len(X)
    if n == 0:
        return X, X, y, y

    # Choose a cutoff timestamp based on sorted timestamps.
    # Supports both DatetimeIndex and MultiIndex(ts_utc, location_id).
    if isinstance(X.index, pd.MultiIndex):
        ts_vals = pd.to_datetime(X.index.get_level_values(0), utc=True)
    else:
        ts_vals = pd.to_datetime(X.index, utc=True)
    idx_sorted = ts_vals.sort_values()
    split_idx = max(1, int(n * (1 - valid_fraction)))
    split_idx = min(split_idx, n - 1) if n > 1 else 0
    cutoff = idx_sorted[split_idx]

    train_mask = ts_vals <= cutoff
    valid_mask = ~train_mask
    X_tr, X_va = X.loc[train_mask], X.loc[valid_mask]
    y_tr, y_va = y.loc[train_mask], y.loc[valid_mask]
    return X_tr, X_va, y_tr, y_va


def _choose_classifier():
    return GradientBoostingClassifier(random_state=42)


def _choose_regressor():
    return GradientBoostingRegressor(random_state=42)


def train_models(cfg: TrainConfig) -> dict:
    """Train rain probability classifier and rain amount regressor.

    Trains models for the maximum horizon provided and logs metrics on a
    time-ordered validation split.

    Returns metadata including features, period, and metrics.
    """
    raw = _load_observations(cfg.db_url, cfg.location_ids, cfg.start_dt, cfg.end_dt)
    logging.info(f"Loaded rows: {len(raw)} across {len(set(raw['location_id']))} locations")

    # For hourly data, use the largest mapping horizon (e.g., 2h for 120 minutes)
    horizon_min = max(cfg.horizons_min)

    X, y_clf, y_reg = _build_supervised(raw, horizon_min)
    logging.info(f"Rows after feature engineering/alignment: {len(X)}")

    # Split by time
    Xc_tr, Xc_va, yc_tr, yc_va = _time_split(X, y_clf, cfg.valid_fraction)
    logging.info(f"Train size (clf): {len(Xc_tr)} | Valid size (clf): {len(Xc_va)}")

    clf = _choose_classifier()
    clf.fit(Xc_tr, yc_tr)
    yc_pred = clf.predict(Xc_va)
    yc_proba = None
    try:
        yc_proba = clf.predict_proba(Xc_va)[:, 1]
    except Exception:
        pass

    # Regressor: train only on rows where future rain occurs
    y_reg_tr = y_reg.loc[Xc_tr.index]
    y_reg_va = y_reg.loc[Xc_va.index]
    rain_mask_tr = (y_reg_tr > 0.0)
    rain_mask_va = (y_reg_va > 0.0)

    Xr_tr = Xc_tr.loc[rain_mask_tr]
    yr_tr = y_reg_tr.loc[rain_mask_tr]
    Xr_va = Xc_va.loc[rain_mask_va]
    yr_va = y_reg_va.loc[rain_mask_va]

    reg = _choose_regressor()
    if len(Xr_tr):
        reg.fit(Xr_tr, yr_tr)
        yr_pred = reg.predict(Xr_va) if len(Xr_va) else np.array([])
    else:
        yr_pred = np.array([])

    # Metrics
    # Compute RMSE via sqrt(MSE) for broader compatibility
    mse_val = float(mean_squared_error(yr_va, yr_pred)) if len(yr_va) and len(yr_pred) else float("nan")
    metrics = {
        "clf_accuracy": float(accuracy_score(yc_va, yc_pred)) if len(yc_va) else float("nan"),
        "clf_roc_auc": float(roc_auc_score(yc_va, yc_proba)) if (yc_proba is not None and len(yc_va) >= 2 and len(set(yc_va)) > 1) else float("nan"),
        "reg_mae_raining": float(mean_absolute_error(yr_va, yr_pred)) if len(yr_va) and len(yr_pred) else float("nan"),
        "reg_rmse_raining": (mse_val ** 0.5) if not np.isnan(mse_val) else float("nan"),
        "horizon_minutes": horizon_min,
    }
    logging.info(f"Validation metrics: {metrics}")

    # Save artifacts
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    clf_path = os.path.join(cfg.artifacts_dir, "rain_clf.joblib")
    reg_path = os.path.join(cfg.artifacts_dir, "rain_reg.joblib")
    meta_path = os.path.join(cfg.artifacts_dir, "metadata.json")
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)

    # Build rich metadata per requirements
    created_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    model_version = created_at_utc
    raw_required = ["temp_c", "humidity", "pressure_hpa", "wind_ms", "precip_mm"]
    raw_optional = ["wind_gust_ms", "cloud_cover"]
    raw_columns_expected = [c for c in raw_required + raw_optional if c in raw.columns]

    metadata = {
        "model_name": "stormcast",
        "global_model": True,
        "model_version": model_version,
        "created_at_utc": created_at_utc,
        "horizons_min": list(cfg.horizons_min),
        "raw_columns_expected": raw_columns_expected,
        "feature_columns": list(X.columns),
        "training_period": {
            "start": cfg.start_dt.isoformat(),
            "end": cfg.end_dt.isoformat(),
        },
        "target_definition": (
            "rain = precip_mm > 0.0; "
            "regressor target = precip_mm shifted forward by horizon steps, trained on raining rows only; "
            "inference may report expected precip as p_rain * E[precip_mm | rain]"
        ),
        "metrics_summary": metrics,
        # Back-compat keys for consumers
        "features": list(X.columns),
        "metrics": metrics,
        "locations": cfg.location_ids,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved classifier to: {clf_path}")
    logging.info(f"Saved regressor to: {reg_path}")
    logging.info(f"Saved metadata to: {meta_path}")

    return metadata


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train rain probability and amount models")
    p.add_argument("--db-url", required=True, help="SQLAlchemy database URL")
    p.add_argument("--location-ids", required=True, help="Comma-separated location IDs")
    p.add_argument("--start", required=True, help="Start datetime ISO (UTC)")
    p.add_argument("--end", required=True, help="End datetime ISO (UTC)")
    p.add_argument(
        "--horizons",
        default="15,30,60,120",
        help="Comma-separated minute horizons (mapped to hours for hourly data)",
    )
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--artifacts-dir", default=os.path.join("ml_engine", "artifacts"), help="Directory to write model artifacts")
    args = p.parse_args()

    location_ids = [int(x) for x in args.location_ids.split(",") if x.strip()]
    horizons_min = [int(x) for x in args.horizons.split(",") if x.strip()]

    def to_utc(s: str) -> dt.datetime:
        s = s.strip()
        # Allow common UTC suffix 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        d = dt.datetime.fromisoformat(s)
        # Treat naive timestamps as UTC
        return d if d.tzinfo is not None else d.replace(tzinfo=dt.timezone.utc)

    cfg = TrainConfig(
        db_url=args.db_url,
        location_ids=location_ids,
        start_dt=to_utc(args.start),
        end_dt=to_utc(args.end),
        horizons_min=horizons_min,
        valid_fraction=args.valid_fraction,
        artifacts_dir=args.artifacts_dir,
    )
    return cfg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _parse_args()
    meta = train_models(cfg)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
