from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime, Float, Integer


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for ingestion models."""


class Observation(Base):
    """Hourly weather observation stored per location.

    Composite primary key: (location_id, ts_utc)
    All measurements are expected to be in standardized units:
    - temp_c: Celsius
    - humidity: percent (0-100)
    - pressure_hpa: hectopascals
    - wind_ms: meters per second
    - wind_gust_ms: meters per second (optional)
    - precip_mm: millimeters
    - cloud_cover: percent (optional)
    """

    __tablename__ = "observations"

    location_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts_utc: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), primary_key=True)

    temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    humidity: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_hpa: Mapped[float] = mapped_column(Float, nullable=False)
    wind_ms: Mapped[float] = mapped_column(Float, nullable=False)
    wind_gust_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_mm: Mapped[float] = mapped_column(Float, nullable=False)
    cloud_cover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


def create_tables(engine) -> None:
    """Create all ingestion tables if they do not exist.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        SQLAlchemy engine for the target database.
    """
    Base.metadata.create_all(bind=engine)
