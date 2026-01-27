from __future__ import annotations

import argparse
import datetime as dt
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd
from sqlalchemy import create_engine

from ml_engine.ingestion.models import create_tables
from ml_engine.ingestion.storage import upsert_hourly_observations


@dataclass(frozen=True)
class Location:
    id: int
    name: str
    lat: float
    lon: float


DEFAULT_LOCATIONS: List[Location] = [
    Location(1, "Brighton", 50.8225, -0.1372),
    Location(2, "London", 51.5074, -0.1278),
    Location(3, "Manchester", 53.4808, -2.2426),
    Location(4, "Birmingham", 52.4862, -1.8904),
    Location(5, "Glasgow", 55.8642, -4.2518),
]


def load_locations_csv(path: str) -> List[Location]:
    """Load locations from a CSV file.

    Expected header columns: id,name,lat,lon
    """
    locations: List[Location] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "name", "lat", "lon"}
        if not reader.fieldnames or not required.issubset({c.strip() for c in reader.fieldnames}):
            raise ValueError(f"locations CSV must include header columns: {sorted(required)}")
        for row in reader:
            if not row:
                continue
            loc = Location(
                id=int(str(row.get("id", "")).strip()),
                name=str(row.get("name", "")).strip(),
                lat=float(str(row.get("lat", "")).strip()),
                lon=float(str(row.get("lon", "")).strip()),
            )
            locations.append(loc)
    if not locations:
        raise ValueError("locations CSV contained no rows")
    return locations


def _to_utc(dt_like: str | dt.datetime) -> dt.datetime:
    d = pd.to_datetime(dt_like, utc=True)
    # pandas returns Timestamp; convert to python datetime with tzinfo UTC
    return d.to_pydatetime()


def meteostat_hourly_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Map Meteostat Hourly DataFrame to the raw schema used by ml_engine.

    Input columns considered (if present):
    - temp -> temp_c (C)
    - rhum -> humidity (%)
    - pres -> pressure_hpa (hPa)
    - prcp -> precip_mm (mm)
    - wspd (km/h) -> wind_ms (m/s)
    - wpgt (km/h) -> wind_gust_ms (m/s, optional)
    - cldc -> cloud_cover (%) (optional, if available)

    The index is converted to a UTC-aware DatetimeIndex and sorted. Rows where
    all mapped columns are NaN are dropped.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    idx = pd.to_datetime(df.index, utc=True)
    df = df.copy()
    df.index = idx

    out = pd.DataFrame(index=df.index)

    # Direct mappings
    direct_map: Dict[str, str] = {
        "temp": "temp_c",
        "rhum": "humidity",
        "pres": "pressure_hpa",
        "prcp": "precip_mm",
    }
    for src, dst in direct_map.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce").astype(float)

    # Wind conversions km/h -> m/s
    if "wspd" in df.columns:
        out["wind_ms"] = pd.to_numeric(df["wspd"], errors="coerce").astype(float) / 3.6
    if "wpgt" in df.columns:
        out["wind_gust_ms"] = pd.to_numeric(df["wpgt"], errors="coerce").astype(float) / 3.6

    # Cloud cover if available
    if "cldc" in df.columns:
        out["cloud_cover"] = pd.to_numeric(df["cldc"], errors="coerce").astype(float)

    # Ensure required field precip_mm is non-null: treat missing as 0
    if "precip_mm" in out.columns:
        out["precip_mm"] = out["precip_mm"].fillna(0.0)

    # Drop rows that are completely empty across mapped columns
    if len(out.columns) > 0:
        out = out.dropna(how="all")

    # Sort and deduplicate index
    out = out[~out.index.duplicated()].sort_index()
    return out


def extract_station_id(stations_df: pd.DataFrame) -> Optional[str]:
    """Extract the first station id from a Meteostat Stations.fetch() result.

    Supports both layouts where the station id is provided as a column or as
    the DataFrame index. Returns None when not available.
    """
    if stations_df is None or len(stations_df) == 0:
        return None
    # Prefer explicit 'id' column if present
    if "id" in stations_df.columns and pd.notna(stations_df["id"].iloc[0]):
        return str(stations_df["id"].iloc[0])
    # Fallback to index as id
    try:
        idx0 = stations_df.index[0]
        if pd.notna(idx0):
            return str(idx0)
    except Exception:
        pass
    return None


def import_meteostat(
    db_url: str,
    start: dt.datetime,
    end: dt.datetime,
    location_ids: Optional[Sequence[int]] = None,
    locations: Optional[Sequence[Location]] = None,
) -> None:
    """Download Meteostat hourly data for selected locations and upsert into DB.

    Parameters
    ----------
    db_url : str
        SQLAlchemy database URL (e.g., sqlite:///stormcast.db).
    start : datetime
        Start datetime (UTC-aware preferred).
    end : datetime
        End datetime (UTC-aware preferred).
    location_ids : Optional[Sequence[int]]
        Subset of location IDs to import (default: all locations).
    locations : Optional[Sequence[Location]]
        Optional explicit locations to use instead of DEFAULT_LOCATIONS.
    """
    # Lazy import to avoid hard dependency for unit tests; use lowercase Meteostat APIs
    try:
        import meteostat as ms  # type: ignore
    except Exception as e:
        raise ImportError("Meteostat is required. Install with: pip install meteostat") from e

    if not hasattr(ms, "stations"):
        raise ImportError("Unsupported Meteostat version: stations API not found")

    def _iter_chunks(start_dt: dt.datetime, end_dt: dt.datetime, days: int = 30):
        cur = start_dt
        while cur < end_dt:
            nxt = min(end_dt, cur + dt.timedelta(days=days))
            yield cur, nxt
            cur = nxt

    def fetch_hourly_df(station_id: str, start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
        h = ms.hourly(station_id, start_dt, end_dt, timezone="UTC")
        df = h.fetch() if hasattr(h, "fetch") else h
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return df

    engine = create_engine(db_url, future=True)
    create_tables(engine)

    pool = list(locations) if locations is not None else DEFAULT_LOCATIONS
    selected: List[Location] = [
        loc for loc in pool
        if location_ids is None or loc.id in set(location_ids)
    ]

    for loc in selected:
        # Find nearest station for the location
        try:
            stations_obj = ms.stations() if callable(ms.stations) else ms.stations
            point = ms.Point(loc.lat, loc.lon)
            stations_df = stations_obj.nearby(point, limit=1)
        except Exception as e:
            print(f"[{loc.id} {loc.name}] Failed to query stations: {e}")
            continue
        station_id = extract_station_id(stations_df)
        if not station_id:
            print(f"[{loc.id} {loc.name}] No station found near ({loc.lat}, {loc.lon}).")
            continue

        total = 0
        imported_min = None
        imported_max = None

        # Large ranges: fetch in 30-day chunks to keep memory and API responses manageable
        for chunk_start, chunk_end in _iter_chunks(start, end, days=30):
            data = fetch_hourly_df(station_id, chunk_start, chunk_end)
            raw = meteostat_hourly_to_raw(data)
            if raw.empty:
                continue
            n = upsert_hourly_observations(engine, raw, location_id=loc.id)
            total += int(n)
            imported_min = raw.index.min() if imported_min is None else min(imported_min, raw.index.min())
            imported_max = raw.index.max() if imported_max is None else max(imported_max, raw.index.max())

        if total == 0:
            print(f"[{loc.id} {loc.name}] Station {station_id}: No data to import.")
            continue
        print(
            f"[{loc.id} {loc.name}] Station {station_id}: Imported {total} rows "
            f"from {imported_min} to {imported_max}"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import Meteostat hourly weather into DB")
    p.add_argument("--db-url", required=True, help="SQLAlchemy URL, e.g., sqlite:///stormcast.db")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD or ISO)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD or ISO)")
    p.add_argument(
        "--location-ids",
        default="",
        help="Comma-separated location IDs to import (default: all)",
    )
    p.add_argument(
        "--locations-csv",
        default="",
        help="Optional CSV file with columns id,name,lat,lon to define locations (default: built-in list)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ids = [int(x) for x in args.location_ids.split(",") if x.strip()] or None
    start = _to_utc(args.start)
    end = _to_utc(args.end)
    locs = load_locations_csv(args.locations_csv) if args.locations_csv else None
    import_meteostat(args.db_url, start, end, ids, locations=locs)


if __name__ == "__main__":
    main()
