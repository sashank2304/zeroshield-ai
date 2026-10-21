from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Exact project features, in the existing prepared/scaled feature space")
    attack_family: str | None = None
