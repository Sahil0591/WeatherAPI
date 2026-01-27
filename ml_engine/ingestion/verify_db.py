from __future__ import annotations

import argparse
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from ml_engine.ingestion.models import Observation, create_tables


def _parse_int_list(csv: str) -> Optional[List[int]]:
    items = [x.strip() for x in (csv or "").split(",") if x.strip()]
    return [int(x) for x in items] if items else None


def fetch_db_stats(db_url: str, location_ids: Optional[Iterable[int]] = None) -> Tuple[int, pd.DataFrame]:
    """Return total observation count and per-location stats."""
    engine = create_engine(db_url, future=True)
    create_tables(engine)

    with Session(engine) as session:
        cond = True
        if location_ids is not None:
            cond = Observation.location_id.in_(list(location_ids))

        total_stmt = select(func.count()).select_from(Observation)
        if location_ids is not None:
            total_stmt = total_stmt.where(cond)
        total = int(session.execute(total_stmt).scalar_one())

        stats_stmt = (
            select(
                Observation.location_id.label("location_id"),
                func.count().label("rows"),
                func.min(Observation.ts_utc).label("min_ts_utc"),
                func.max(Observation.ts_utc).label("max_ts_utc"),
            )
            .select_from(Observation)
        )
        if location_ids is not None:
            stats_stmt = stats_stmt.where(cond)
        stats_stmt = stats_stmt.group_by(Observation.location_id).order_by(func.count().desc())

        rows = session.execute(stats_stmt).all()

    df = pd.DataFrame(rows, columns=["location_id", "rows", "min_ts_utc", "max_ts_utc"])
    return total, df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify stormcast DB coverage (row counts and date ranges)")
    p.add_argument("--db-url", required=True, help="SQLAlchemy URL, e.g., sqlite:///stormcast.db")
    p.add_argument(
        "--location-ids",
        default="",
        help="Optional comma-separated location IDs to inspect (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ids = _parse_int_list(args.location_ids)

    total, stats = fetch_db_stats(args.db_url, ids)

    print(f"Total observations: {total}")
    if stats.empty:
        print("No observations found.")
        return

    # Print a compact table
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
