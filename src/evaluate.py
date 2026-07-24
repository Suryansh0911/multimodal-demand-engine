import torch
import yaml
import glob
import numpy as np
from tfrecord.torch.dataset import TFRecordDataset
from src.models.multimodal_fusion import MultimodalDemandEngine

def run_inference():
    # 1. Load Config
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultimodalDemandEngine(config)
    checkpoint_path = "models/checkpoints/multimodal_demand_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Checkpoint loaded successfully!")

    tfrecord_files = sorted(glob.glob("data/processed/*.tfrecord"))
    description = {
        "text": "byte",
        "tabular_features": "float",
        "historical_sales": "float",
        "future_demand": "float",
    }
    
    dataset = TFRecordDataset(tfrecord_files[0], index_path=None, description=description)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5)

    with torch.no_grad():
        for batch in loader:
            bs = len(batch["future_demand"])
            
            input_ids = torch.ones((bs, 128), dtype=torch.long, device=device) * 101
            attention_mask = torch.ones((bs, 128), dtype=torch.long, device=device)
            tabular_features = batch["tabular_features"].to(device).float()
            
            historical_sales = batch["historical_sales"].to(device).float()
            historical_sales = (historical_sales - historical_sales.mean(dim=-1, keepdim=True)) / (historical_sales.std(dim=-1, keepdim=True) + 1e-6)

            actual_raw_demand = batch["future_demand"].numpy().flatten()

            log_preds = model(input_ids, attention_mask, tabular_features, historical_sales)
            
            real_demand_preds = torch.expm1(log_preds).cpu().numpy()

            print("\n================ EVALUATION SAMPLE ================")
            for i in range(bs):
                print(f"Sample {i+1}:")
                print(f"  --> Actual Demand:    {actual_raw_demand[i]:.2f} units")
                print(f"  --> Predicted Demand: {real_demand_preds[i]:.2f} units")
                print(f"  --> Error:            {abs(actual_raw_demand[i] - real_demand_preds[i]):.2f} units\n")
            break

if __name__ == "__main__":
    run_inference()