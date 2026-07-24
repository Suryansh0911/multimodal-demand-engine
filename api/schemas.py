from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    text_input_ids: List[int] = Field(..., min_length=128, max_length=128)
    temporal_history: List[float] = Field(..., min_length=30, max_length=30)
    tabular_features: List[float] = Field(..., min_length=15, max_length=15)

class PredictionResponse(BaseModel):
    predicted_demand: float
    latency_ms: float
    model_version: str