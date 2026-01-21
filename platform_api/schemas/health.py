from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_env: str
    app_name: str
