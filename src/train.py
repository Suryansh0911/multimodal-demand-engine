import glob
import os
import torch
import torch.nn as nn
import yaml
from tfrecord.torch.dataset import TFRecordDataset
from src.models.multimodal_fusion import MultimodalDemandEngine


def main():
    config_path = "configs/train_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Build Model
    model = MultimodalDemandEngine(config).to(device)

    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 0.0005))
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 4))

    # Dataset Files
    tfrecord_pattern = config.get("dataset", {}).get("tfrecords_pattern", "data/processed/*.tfrecord")
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))

    if not tfrecord_files:
        raise FileNotFoundError(f"Expected .tfrecord files matching {tfrecord_pattern}, but found none!")

    print(f"✅ Found {len(tfrecord_files)} .tfrecord files!")

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("mixed_precision", True))

    description = {
        "text": "byte",
        "tabular_features": "float",
        "historical_sales": "float",
        "future_demand": "float",
    }

    model.train()
    print("⚡ Starting PyTorch Training Loop...\n")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_batches = 0

        for tf_file in tfrecord_files:
            dataset = TFRecordDataset(tf_file, index_path=None, description=description)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)

            for batch in loader:
                bs = len(batch["future_demand"])
                input_ids = torch.ones((bs, 128), dtype=torch.long, device=device) * 101
                attention_mask = torch.ones((bs, 128), dtype=torch.long, device=device)

                tabular_features = batch["tabular_features"].to(device).float()
                
                historical_sales = batch["historical_sales"].to(device).float()
                historical_sales = (historical_sales - historical_sales.mean(dim=-1, keepdim=True)) / (historical_sales.std(dim=-1, keepdim=True) + 1e-6)

                raw_targets = batch["future_demand"].to(device).float().reshape(-1)
                targets = torch.log1p(torch.clamp(raw_targets, min=0.0))

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=train_cfg.get("mixed_precision", True)):
                    predictions = model(input_ids, attention_mask, tabular_features, historical_sales)
                    predictions = predictions.reshape(-1)
                    loss = criterion(predictions, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                total_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(total_batches, 1)
        print(f"Epoch [{epoch}/{epochs}] - Log-MSE Loss: {epoch_loss:.4f} - Log-RMSE: {epoch_loss**0.5:.4f}")

    # Save Checkpoint
    save_dir = train_cfg.get("model_dir", "models/checkpoints/")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "multimodal_demand_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()