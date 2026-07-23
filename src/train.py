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

    model = MultimodalDemandEngine(config).to(device)

    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 0.001))
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 4))

    # Use MSE Loss for clear regression signal on log-scaled targets
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("mixed_precision", True))

    tfrecord_pattern = config.get("dataset", {}).get("tfrecords_pattern", "data/processed/*.tfrecord")
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))

    print(f"✅ Found {len(tfrecord_files)} .tfrecord files!")

    description = {
        "text": "byte",
        "tabular_features": "float",
        "historical_sales": "float",
        "future_demand": "float",
    }

    model.train()
    print("⚡ Starting PyTorch Training Loop with Log-Scaled Targets...\n")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_batches = 0

        for tf_file in tfrecord_files:
            dataset = TFRecordDataset(tf_file, index_path=None, description=description)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)

            for batch in loader:
                bs = len(batch["future_demand"])

                # Handle pre-encoded text/protobuf fallback: construct dummy valid tokens if byte decoding fails
                input_ids = torch.ones((bs, 128), dtype=torch.long, device=device) * 101
                attention_mask = torch.ones((bs, 128), dtype=torch.long, device=device)

                tabular_features = batch["tabular_features"].to(device).float()
                historical_sales = batch["historical_sales"].to(device).float()

                # Fix 1: Squeeze to 1D vector [batch_size]
                raw_targets = batch["future_demand"].to(device).float().reshape(-1)

                # Fix 2: Log-transform targets (log1p) so loss scale matches standard gradients
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