import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import yaml
from src.data.pipeline import DemandDataset, build_dataloader
from src.models.multimodal_fusion import MultimodalDemandEngine


def main():
    # Load configuration
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Build Model
    model = MultimodalDemandEngine(config).to(device)

    # Loss & Optimizer
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    scaler = GradScaler(
        enabled=config["training"].get("mixed_precision", True)
    )

    # Mock/Processed Dataset Setup (Replace with actual loaded features)
    dummy_data = [
        {
            "input_ids": [101] + [1000] * 126 + [102],
            "attention_mask": [1] * 128,
            "tabular_features": [0.5] * 10,
            "target": 12.5,
        }
        for _ in range(200)
    ]

    train_dataset = DemandDataset(dummy_data)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        is_training=True,
    )

    # Training Loop
    epochs = config["training"]["epochs"]
    model.train()

    print("⚡ Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tabular_features = batch["tabular_features"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            # Automatic Mixed Precision (AMP)
            with autocast(
                enabled=config["training"].get("mixed_precision", True)
            ):
                predictions = model(input_ids, attention_mask, tabular_features)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch [{epoch}/{epochs}] - Loss (Huber): {epoch_loss:.4f} - RMSE: {epoch_loss**0.5:.4f}"
        )

    # Save Checkpoint
    save_dir = config["training"].get("model_dir", "models/checkpoints/")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "multimodal_demand_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()