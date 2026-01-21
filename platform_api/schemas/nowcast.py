from typing import Optional

from pydantic import BaseModel, Field


class NowcastRequest(BaseModel):
    location: str
    timestamp: Optional[str] = None


class NowcastResponse(BaseModel):
    location: str
    timestamp: str
    precipitation: float = Field(ge=0)
    windspeed: float = Field(ge=0)
