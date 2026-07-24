from fastapi import FastAPI
from api.schemas import PredictionRequest, PredictionResponse
from api.inference import TFLiteModelRunner
import time
import numpy as np

app = FastAPI(title="Multimodal Demand Forecasting API", version="1.0.0")

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    start_time = time.time()
    vision_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    text_input = np.array([request.text_input_ids], dtype=np.int32)
    temporal_input = np.array([request.temporal_history], dtype=np.float32)
    tabular_input = np.array([request.tabular_features], dtype=np.float32)

    prediction = 142.5  # Mock output
    
    latency = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        predicted_demand=round(prediction, 2),
        latency_ms=round(latency, 2),
        model_version="1.0-quantized"
    )