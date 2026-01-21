from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field()
    uptime_s: float = Field(ge=0)
    version: str = Field()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "ok", "uptime_s": 12.34, "version": "0.1.0"}
            ]
        }
    }
