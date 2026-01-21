from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Observation


def _ensure_utc(dt_like: dt.datetime) -> dt.datetime:
    if dt_like.tzinfo is None:
        return dt_like.replace(tzinfo=dt.timezone.utc)
    return dt_like.astimezone(dt.timezone.utc)


def upsert_hourly_observations(engine, df: pd.DataFrame, location_id: int) -> int:
    """Insert or update hourly observations into the database.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        Database engine.
    df : pandas.DataFrame
        DataFrame indexed by UTC datetime. Required columns:
        `temp_c`, `humidity`, `pressure_hpa`, `wind_ms`, `precip_mm`.
        Optional columns: `wind_gust_ms`, `cloud_cover`.
    location_id : int
        Identifier for the observation location.

    Returns
    -------
    int
        Number of rows inserted/updated.

    Notes
    -----
    - Missing optional columns are stored as NULL.
    - Upsert strategy is naive but portable: fetch existing row then update, or
      insert when missing. Suitable for small to moderate batch sizes.
    """
    required = ["temp_c", "humidity", "pressure_hpa", "wind_ms", "precip_mm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    count = 0
    with Session(engine) as session:
        for ts, row in df.iterrows():
            ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            ts_utc = _ensure_utc(ts_py)

            existing = session.get(Observation, (location_id, ts_utc))
            payload = {
                "temp_c": float(row.get("temp_c")) if pd.notna(row.get("temp_c")) else None,
                "humidity": float(row.get("humidity")) if pd.notna(row.get("humidity")) else None,
                "pressure_hpa": float(row.get("pressure_hpa")) if pd.notna(row.get("pressure_hpa")) else None,
                "wind_ms": float(row.get("wind_ms")) if pd.notna(row.get("wind_ms")) else None,
                "precip_mm": float(row.get("precip_mm")) if pd.notna(row.get("precip_mm")) else None,
                "wind_gust_ms": float(row.get("wind_gust_ms")) if "wind_gust_ms" in df.columns and pd.notna(row.get("wind_gust_ms")) else None,
                "cloud_cover": float(row.get("cloud_cover")) if "cloud_cover" in df.columns and pd.notna(row.get("cloud_cover")) else None,
            }

            if existing is None:
                obs = Observation(
                    location_id=location_id,
                    ts_utc=ts_utc,
                    **payload,
                )
                session.add(obs)
            else:
                for k, v in payload.items():
                    setattr(existing, k, v)
            count += 1
        session.commit()
    return count


def read_recent_observations(engine, location_id: int, hours_back: int) -> pd.DataFrame:
    """Read recent hourly observations for a location.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        Database engine.
    location_id : int
        Location identifier.
    hours_back : int
        Number of hours to look back from current UTC time (inclusive).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by UTC datetime with columns:
        `temp_c`, `humidity`, `pressure_hpa`, `wind_ms`, `precip_mm`, and optional
        `wind_gust_ms`, `cloud_cover` (NaN when missing).
    """
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_back)
    with Session(engine) as session:
        stmt = (
            select(Observation)
            .where(Observation.location_id == location_id)
            .where(Observation.ts_utc >= cutoff)
            .order_by(Observation.ts_utc)
        )
        rows = session.execute(stmt).scalars().all()

    if not rows:
        return pd.DataFrame(
            columns=[
                "temp_c",
                "humidity",
                "pressure_hpa",
                "wind_ms",
                "precip_mm",
                "wind_gust_ms",
                "cloud_cover",
            ]
        ).set_index(pd.DatetimeIndex([], tz="UTC"))

    index = pd.to_datetime([r.ts_utc for r in rows], utc=True)
    data = {
        "temp_c": [r.temp_c for r in rows],
        "humidity": [r.humidity for r in rows],
        "pressure_hpa": [r.pressure_hpa for r in rows],
        "wind_ms": [r.wind_ms for r in rows],
        "precip_mm": [r.precip_mm for r in rows],
        "wind_gust_ms": [r.wind_gust_ms for r in rows],
        "cloud_cover": [r.cloud_cover for r in rows],
    }
    df = pd.DataFrame(data, index=index)
    return df
