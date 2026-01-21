from typing import List, Optional

from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    location: str
    timestamp: Optional[str] = None


class FeatureImportance(BaseModel):
    name: str
    importance: float = Field(ge=0)


class ExplainResponse(BaseModel):
    location: str
    timestamp: str
    top_features: List[FeatureImportance]
