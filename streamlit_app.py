import os
import urllib.request
import yaml
import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer
from src.models.multimodal_fusion import MultimodalDemandEngine

st.set_page_config(
    page_title="Multimodal Demand Engine",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multimodal Demand Engine")
st.markdown("Forecast future product demand using Product Description, Tabular Features, and Historical Sales Sequences.")

@st.cache_resource
def load_model_and_tokenizer():
    config_path = "configs/train_config.yaml"
    checkpoint_dir = "models/checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "multimodal_demand_model.pt")

    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        with st.spinner("Downloading model weights from cloud storage..."):
            MODEL_URL = "https://huggingface.co/Ryan911/multimodal-demand-engine/resolve/main/multimodal_demand_model.pt"
            
            try:
                urllib.request.urlretrieve(MODEL_URL, checkpoint_path)
                st.toast("✅ Model checkpoint downloaded successfully!")
            except Exception as download_err:
                st.error(f"Failed to download model weights: {download_err}")
                st.stop()

    if not os.path.exists(config_path):
        st.error(f"Config missing at {config_path}")
        st.stop()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")

    hf_path = config.get("model", {}).get("local_hf_path", "distilbert-base-uncased")
    if not os.path.exists(hf_path):
        hf_path = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(hf_path)

    model = MultimodalDemandEngine(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model, tokenizer, config

try:
    model, tokenizer, config = load_model_and_tokenizer()
    st.success("✅ Model and DistilBERT successfully loaded!")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Text Context")
    product_description = st.text_area(
        "Product Description",
        value="Premium wireless noise-canceling headphones with spatial audio and long battery life.",
        height=100
    )

    st.subheader("2. Tabular Features (15 Metrics)")
    st.caption("Provide values for the 15 tabular features (e.g., price, discount, rating):")
    
    default_tabular = [0.12, 0.45, 0.88, 0.02, 0.15, 0.90, 0.33, 0.41, 0.05, 0.11, 0.22, 0.34, 0.55, 0.61, 0.72]
    tabular_inputs = []
    
    tab_cols = st.columns(3)
    for i in range(15):
        with tab_cols[i % 3]:
            val = st.number_input(f"Feature {i+1}", value=default_tabular[i], key=f"tab_{i}")
            tabular_inputs.append(val)

with col2:
    st.subheader("3. Historical Sales Sequence (30 Days)")
    st.caption("Input or edit the historical sales volume across the past 30 days:")

    default_sales = [
        12.0, 15.2, 14.0, 18.5, 20.1, 19.0, 22.4, 21.0, 25.0, 24.1, 
        23.0, 28.4, 30.1, 29.5, 31.0, 35.2, 34.0, 38.0, 36.5, 40.0, 
        42.1, 41.0, 45.0, 44.2, 48.0, 46.5, 50.1, 49.0, 52.3, 54.0
    ]

    edited_sales = st.data_editor(
        {"Day": [f"Day {i+1}" for i in range(30)], "Sales Volume": default_sales},
        num_rows="fixed",
        height=380,
        use_container_width=True
    )
    historical_sales_inputs = edited_sales["Sales Volume"]

st.divider()

if st.button("🚀 Forecast Demand", type="primary", use_container_width=True):
    with st.spinner("Running PyTorch Multimodal Inference..."):
        try:
            encoded = tokenizer(
                product_description,
                padding="max_length",
                truncation=True,
                max_length=config.get("dataset", {}).get("sequence_length", 128),
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            tab_feat = torch.tensor([tabular_inputs], dtype=torch.float32)

            hist_sales = torch.tensor([historical_sales_inputs], dtype=torch.float32)
            hist_sales = (hist_sales - hist_sales.mean(dim=-1, keepdim=True)) / (hist_sales.std(dim=-1, keepdim=True) + 1e-6)

            with torch.no_grad():
                log_pred = model(input_ids, attention_mask, tab_feat, hist_sales).item()
                real_demand = max(0.0, float(np.expm1(log_pred)))

            st.success("✅ Inference completed successfully!")
            st.metric(
                label="Predicted Future Demand Volume",
                value=f"{real_demand:.2f} Units",
                delta=f"{real_demand - historical_sales_inputs[-1]:+.2f} Units relative to last day"
            )

            chart_data = list(historical_sales_inputs) + [real_demand]
            
            st.line_chart({"Sales / Forecast": chart_data}, use_container_width=True)

        except Exception as e:
            st.error(f"Inference Error: {str(e)}")