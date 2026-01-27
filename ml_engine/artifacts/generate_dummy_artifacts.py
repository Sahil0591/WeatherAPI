import os
import json
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
import joblib

from ml_engine.features.build_features import build_features


def make_synthetic_raw(n_hours: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.utcnow().floor('h')
    idx = pd.date_range(end - pd.Timedelta(hours=n_hours-1), end, freq='h', tz='UTC')

    # Base signals
    temp = 15 + 10*np.sin(np.linspace(0, 3*np.pi, n_hours)) + rng.normal(0, 2, n_hours)
    hum = 60 + 20*np.sin(np.linspace(0, 2*np.pi, n_hours)) + rng.normal(0, 5, n_hours)
    press = 1013 + 5*np.sin(np.linspace(0, 4*np.pi, n_hours)) + rng.normal(0, 1.5, n_hours)
    wind = np.abs(3 + 2*np.sin(np.linspace(0, 6*np.pi, n_hours)) + rng.normal(0, 1, n_hours))

    # Rain events: random sparse spikes
    rain_events = rng.choice([0, 1], size=n_hours, p=[0.85, 0.15])
    precip = np.where(rain_events == 1, np.clip(rng.normal(2.0, 1.0, n_hours), 0, None), 0.0)

    df = pd.DataFrame({
        'temp_c': temp,
        'humidity': hum,
        'pressure_hpa': press,
        'wind_ms': wind,
        'precip_mm': precip,
    }, index=idx)
    return df


def train_on_synthetic(horizon_min: int = 120, artifacts_dir: str = os.path.join('ml_engine', 'artifacts')) -> dict:
    raw = make_synthetic_raw()
    feats = build_features(raw)

    # Align labels: shift future by mapped hour steps
    steps = max(1, (horizon_min + 59)//60)  # ceil hours
    y_future = raw['precip_mm'].shift(-steps)

    aligned = pd.concat([
        feats,
        (y_future > 0.0).astype(int).rename('y_clf'),
        y_future.astype(float).rename('y_reg')
    ], axis=1).dropna(subset=['y_clf', 'y_reg'])

    X = aligned[feats.columns].fillna(0.0)
    y_clf = aligned['y_clf'].astype(int)
    y_reg = aligned['y_reg'].astype(float)

    # Time-based split
    n = len(X)
    split = max(1, int(n*0.8))
    Xc_tr, Xc_va = X.iloc[:split], X.iloc[split:]
    yc_tr, yc_va = y_clf.iloc[:split], y_clf.iloc[split:]

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xc_tr, yc_tr)

    yc_pred = clf.predict(Xc_va)
    try:
        yc_proba = clf.predict_proba(Xc_va)[:, 1]
    except Exception:
        yc_proba = None

    # Regress only on raining rows
    rain_mask_tr = (y_reg.iloc[:len(Xc_tr)] > 0.0)
    rain_mask_va = (y_reg.iloc[len(Xc_tr):] > 0.0)
    Xr_tr, yr_tr = Xc_tr[rain_mask_tr], y_reg.iloc[:len(Xc_tr)][rain_mask_tr]
    Xr_va, yr_va = Xc_va[rain_mask_va], y_reg.iloc[len(Xc_tr):][rain_mask_va]

    reg = GradientBoostingRegressor(random_state=42)
    if len(Xr_tr):
        reg.fit(Xr_tr, yr_tr)
        yr_pred = reg.predict(Xr_va) if len(Xr_va) else np.array([])
    else:
        yr_pred = np.array([])

    mse_val = float(mean_squared_error(yr_va, yr_pred)) if len(yr_va) and len(yr_pred) else float('nan')
    metrics = {
        'clf_accuracy': float(accuracy_score(yc_va, yc_pred)) if len(yc_va) else float('nan'),
        'clf_roc_auc': float(roc_auc_score(yc_va, yc_proba)) if (yc_proba is not None and len(yc_va) >= 2 and len(set(yc_va)) > 1) else float('nan'),
        'reg_mae_raining': float(mean_absolute_error(yr_va, yr_pred)) if len(yr_va) and len(yr_pred) else float('nan'),
        'reg_rmse_raining': (mse_val**0.5) if not np.isnan(mse_val) else float('nan'),
        'horizon_minutes': horizon_min,
    }

    os.makedirs(artifacts_dir, exist_ok=True)
    clf_path = os.path.join(artifacts_dir, 'rain_clf.joblib')
    reg_path = os.path.join(artifacts_dir, 'rain_reg.joblib')
    meta_path = os.path.join(artifacts_dir, 'metadata.json')
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)

    created_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    model_version = created_at_utc
    metadata = {
        'model_name': 'stormcast',
        'model_version': model_version,
        'created_at_utc': created_at_utc,
        'horizons_min': [60, 120],
        'raw_columns_expected': ['temp_c', 'humidity', 'pressure_hpa', 'wind_ms', 'precip_mm'],
        'feature_columns': list(X.columns),
        'training_period': {
            'start': str(X.index.min().isoformat()),
            'end': str(X.index.max().isoformat()),
        },
        'target_definition': 'rain = precip_mm > 0.0; regression target = precip_mm shifted forward by horizon steps',
        'metrics_summary': metrics,
        # Back-compat
        'features': list(X.columns),
        'metrics': metrics,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return {
        'clf_path': clf_path,
        'reg_path': reg_path,
        'meta_path': meta_path,
        'metrics': metrics,
    }


if __name__ == '__main__':
    out = train_on_synthetic()
    print(json.dumps(out, indent=2))
