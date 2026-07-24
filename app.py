import os
import urllib.request
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
    checkpoint_dir = "models/checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "multimodal_demand_model.pt")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Auto-download model checkpoint if missing locally or on cloud server
    if not os.path.exists(checkpoint_path):
        print("⏳ Model checkpoint missing. Auto-downloading from Hugging Face...")
        # ⬇️ REPLACE THIS WITH YOUR DIRECT HF / CLOUD DOWNLOAD URL
        model_url = os.getenv(
            "MODEL_URL",
            "https://huggingface.co/Ryan911/multimodal-demand-engine/resolve/main/multimodal_demand_model.pt"
        )
        
        try:
            urllib.request.urlretrieve(model_url, checkpoint_path)
            print("✅ Checkpoint successfully downloaded!")
        except Exception as err:
            raise RuntimeError(f"Failed to auto-download model checkpoint from {model_url}: {err}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file missing at {config_path}")

    # Load Config
    with open(config_path, "r") as f:
        CONFIG = yaml.safe_load(f)

    # Initialize Tokenizer
    hf_path = CONFIG.get("model", {}).get("local_hf_path", "distilbert-base-uncased")
    if not os.path.exists(hf_path):
        hf_path = "distilbert-base-uncased"
    TOKENIZER = AutoTokenizer.from_pretrained(hf_path)

    # Initialize Model Architecture & Load Saved State
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

    expected_tab_dim = CONFIG.get("model", {}).get("num_tabular_features", 15)
    if len(payload.tabular_features) != expected_tab_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_tab_dim} tabular features, got {len(payload.tabular_features)}."
        )

    try:
        with torch.no_grad():
            encoded = TOKENIZER(
                payload.product_description,
                padding="max_length",
                truncation=True,
                max_length=CONFIG.get("dataset", {}).get("sequence_length", 128),
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)

            tab_feat = torch.tensor([payload.tabular_features], dtype=torch.float32).to(DEVICE)

            hist_sales = torch.tensor([payload.historical_sales], dtype=torch.float32).to(DEVICE)
            hist_sales = (hist_sales - hist_sales.mean(dim=-1, keepdim=True)) / (hist_sales.std(dim=-1, keepdim=True) + 1e-6)

            log_prediction = MODEL(input_ids, attention_mask, tab_feat, hist_sales).item()

            real_demand_units = float(np.expm1(log_prediction))
            real_demand_units = max(0.0, real_demand_units)

        return InferenceResponse(
            status="success",
            predicted_demand_units=round(real_demand_units, 2),
            log_demand_scale=round(float(log_prediction), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")