from __future__ import annotations

from typing import List, Optional

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator


logger = structlog.get_logger()


class NowcastRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    horizon: int = Field(120, gt=0, description="Default forecast horizon in minutes")
    horizons_min: Optional[List[int]] = Field(
        default=None, description="Optional list of specific horizons in minutes"
    )

    @field_validator("horizons_min")
    @classmethod
    def _validate_horizons(cls, v: Optional[List[int]]):
        if v is None:
            return v
        if any(h <= 0 for h in v):
            raise ValueError("All horizons_min entries must be > 0")
        return v


class NowcastPrediction(BaseModel):
    minutes: int = Field(..., gt=0)
    p_rain: float = Field(..., ge=0.0, le=1.0)
    rain_mm: float = Field(..., ge=0.0)
    rain_mm_p10: float = Field(..., ge=0.0)
    rain_mm_p90: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _soft_check_quantiles(self) -> "NowcastPrediction":
        if not (self.rain_mm_p10 <= self.rain_mm <= self.rain_mm_p90):
            logger.warning(
                "prediction_quantiles_inconsistent",
                minutes=self.minutes,
                p10=self.rain_mm_p10,
                value=self.rain_mm,
                p90=self.rain_mm_p90,
            )
        return self


class ModelInfo(BaseModel):
    name: str
    version: str


class Location(BaseModel):
    lat: float
    lon: float


class NowcastResponse(BaseModel):
    location: Location
    generated_at: str
    horizons_min: List[int]
    predictions: List[NowcastPrediction]
    model: ModelInfo


class ExplainFactor(BaseModel):
    name: str
    contribution: float


class ExplainResponse(BaseModel):
    summary: str
    top_factors: List[ExplainFactor]
