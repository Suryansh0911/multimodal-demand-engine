import os
import yaml
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from src.models.multimodal_fusion import MultimodalDemandEngine

# 1. Initialize FastAPI App
app = FastAPI(
    title="Multimodal Demand Engine API",
    description="Inference server for predicting future product demand using multimodal inputs (text, tabular, sequence).",
    version="1.0.0"
)

# 2. Global State Variables
MODEL = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {}


@app.on_event("startup")
def load_artifacts():
    """Loads model weights, tokenizer, and config into GPU/CPU memory on startup."""
    global MODEL, TOKENIZER, CONFIG

    config_path = "configs/train_config.yaml"
    checkpoint_path = "models/checkpoints/multimodal_demand_model.pt"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file missing at {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint missing at {checkpoint_path}")

    # Load Config
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    # Initialize Tokenizer
    hf_path = CONFIG.get("model", {}).get("local_hf_path", "distilbert-base-uncased")
    if not os.path.exists(hf_path):
        hf_path = "distilbert-base-uncased"
    TOKENIZER = AutoTokenizer.from_pretrained(hf_path)

    # Reconstruct Architecture & Load Saved Checkpoint
    MODEL = MultimodalDemandEngine(CONFIG)
    MODEL.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()

    print(f"✅ Multimodal Demand Model successfully loaded on {DEVICE}")


# 3. Input Validation Schema
class InferenceRequest(BaseModel):
    product_description: str = Field(
        default="Premium wireless noise-canceling headphones",
        description="Text description or title of the item."
    )
    tabular_features: list[float] = Field(
        ...,
        description="15-dimensional numerical feature vector (e.g., price, discount, inventory level)."
    )
    historical_sales: list[float] = Field(
        ...,
        description="Historical sales sequence across time steps (e.g., last 30 days)."
    )


class InferenceResponse(BaseModel):
    status: str
    predicted_demand_units: float
    log_demand_scale: float


# 4. API Endpoints
@app.get("/")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None
    }


@app.post("/predict", response_model=InferenceResponse)
def predict_demand(payload: InferenceRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=500, detail="Model runtime not initialized.")

    # Validate input dimensions
    expected_tab_dim = CONFIG.get("model", {}).get("num_tabular_features", 15)
    if len(payload.tabular_features) != expected_tab_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_tab_dim} tabular features, got {len(payload.tabular_features)}."
        )

    try:
        with torch.no_grad():
            # A. Tokenize Text
            encoded = TOKENIZER(
                payload.product_description,
                padding="max_length",
                truncation=True,
                max_length=CONFIG.get("dataset", {}).get("sequence_length", 128),
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)

            # B. Format Tabular Features
            tab_feat = torch.tensor([payload.tabular_features], dtype=torch.float32).to(DEVICE)

            # C. Format & Standardize Historical Sales Sequence
            hist_sales = torch.tensor([payload.historical_sales], dtype=torch.float32).to(DEVICE)
            hist_sales = (hist_sales - hist_sales.mean(dim=-1, keepdim=True)) / (hist_sales.std(dim=-1, keepdim=True) + 1e-6)

            # D. Model Forward Pass
            log_prediction = MODEL(input_ids, attention_mask, tab_feat, hist_sales).item()

            # E. Invert Log-Scale: expm1(y) = e^y - 1
            real_demand_units = float(np.expm1(log_prediction))
            real_demand_units = max(0.0, real_demand_units)  # Clamp negative demand to 0

        return InferenceResponse(
            status="success",
            predicted_demand_units=round(real_demand_units, 2),
            log_demand_scale=round(float(log_prediction), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")