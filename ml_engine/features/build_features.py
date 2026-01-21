from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from raw hourly weather observations.

    Input
    -----
    df : pandas.DataFrame
        Frame indexed by UTC datetime, containing raw columns:
        `temp_c`, `humidity`, `pressure_hpa`, `wind_ms`, `precip_mm` (and possibly others).

    Output
    ------
    pandas.DataFrame
        Hourly feature frame with no missing timestamps (resampled to hourly), sorted,
        and the following columns:
        - precip_lag_1h, precip_lag_3h, precip_lag_6h
        - humidity_mean_3h, pressure_mean_6h, wind_mean_3h
        - pressure_delta_3h, humidity_delta_3h, temp_delta_3h
        - pressure_std_6h
        - hour_of_day, day_of_week, month

    Notes
    -----
    - All features are aligned to use only past data relative to each row.
      Rolling statistics are shifted by 1 hour to exclude the current observation.
    - The function resamples to hourly frequency using asfreq and sorts the index.
    - Required raw columns: `temp_c`, `humidity`, `pressure_hpa`, `wind_ms`, `precip_mm`.
    """
    required = ["temp_c", "humidity", "pressure_hpa", "wind_ms", "precip_mm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime index with UTC tz and hourly continuity
    idx = pd.to_datetime(df.index, utc=True)
    df = df.copy()
    df.index = idx
    df = df.sort_index()
    df = df.resample("h").asfreq()

    # Convenience references
    temp = df["temp_c"].astype(float)
    hum = df["humidity"].astype(float)
    press = df["pressure_hpa"].astype(float)
    wind = df["wind_ms"].astype(float)
    precip = df["precip_mm"].astype(float)

    # Lags (past-only)
    lag1 = precip.shift(1)
    lag3 = precip.shift(3)
    lag6 = precip.shift(6)

    # Rolling means (exclude current via shift)
    hum_mean_3h = hum.rolling(3, min_periods=3).mean().shift(1)
    press_mean_6h = press.rolling(6, min_periods=6).mean().shift(1)
    wind_mean_3h = wind.rolling(3, min_periods=3).mean().shift(1)

    # Deltas (current - value 3 hours ago)
    press_delta_3h = press - press.shift(3)
    hum_delta_3h = hum - hum.shift(3)
    temp_delta_3h = temp - temp.shift(3)

    # Rolling std (exclude current via shift), sample std (ddof=1)
    press_std_6h = press.rolling(6, min_periods=6).std().shift(1)

    # Time features
    hour_of_day = df.index.hour
    day_of_week = df.index.dayofweek
    month = df.index.month

    features = pd.DataFrame(
        {
            "precip_lag_1h": lag1,
            "precip_lag_3h": lag3,
            "precip_lag_6h": lag6,
            "humidity_mean_3h": hum_mean_3h,
            "pressure_mean_6h": press_mean_6h,
            "wind_mean_3h": wind_mean_3h,
            "pressure_delta_3h": press_delta_3h,
            "humidity_delta_3h": hum_delta_3h,
            "temp_delta_3h": temp_delta_3h,
            "pressure_std_6h": press_std_6h,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "month": month,
        },
        index=df.index,
    )

    # Sort and de-duplicate index (safety)
    features = features[~features.index.duplicated()].sort_index()
    return features
