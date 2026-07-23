import glob
import os
import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer
from tfrecord.torch.dataset import TFRecordDataset
from src.models.multimodal_fusion import MultimodalDemandEngine


def main():
    # Load configuration
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Initialize Tokenizer for 'text' feature
    tokenizer_name = config["model"].get("text_backbone", "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Build Model
    model = MultimodalDemandEngine(config).to(device)

    # Optimizer, Scheduler & Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config["training"].get("mixed_precision", True)
    )
    criterion = nn.HuberLoss()

    # Find .tfrecord files
    data_dir = "data/processed"
    tfrecord_files = sorted(glob.glob(f"{data_dir}/*.tfrecord"))

    if not tfrecord_files:
        raise FileNotFoundError(
            f"Expected .tfrecord files in {data_dir}, but found none!"
        )

    print(f"✅ Found {len(tfrecord_files)} .tfrecord files! Mapping schema...")

    # Schema description corresponding strictly to your dataset keys
    description = {
        "text": "byte",
        "tabular_features": "float",
        "historical_sales": "float",
        "future_demand": "float",
    }

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    model.train()

    print("⚡ Starting Training Loop on Real Dataset...\n")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_batches = 0

        for tf_file in tfrecord_files:
            dataset = TFRecordDataset(tf_file, index_path=None, description=description)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

            for batch in loader:
                # 1. Decode byte strings from TFRecord to text
                raw_texts = [
                    t.decode("utf-8", errors="ignore") if isinstance(t, bytes) else str(t)
                    for t in batch["text"]
                ]

                # 2. Dynamic Tokenization
                encoded = tokenizer(
                    raw_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                tabular_features = batch["tabular_features"].to(device)
                targets = batch["future_demand"].to(device)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=config["training"].get("mixed_precision", True)):
                    predictions = model(input_ids, attention_mask, tabular_features)
                    loss = criterion(predictions, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                total_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(total_batches, 1)
        print(f"Epoch [{epoch}/{epochs}] - Huber Loss: {epoch_loss:.4f} - RMSE: {epoch_loss**0.5:.4f}")

    # Save Checkpoint
    save_dir = config["training"].get("model_dir", "models/checkpoints/")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "multimodal_demand_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()