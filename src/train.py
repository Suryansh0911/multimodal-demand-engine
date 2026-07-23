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

    # Tokenizer setup
    model_cfg = config.get("model", {})
    hf_path = model_cfg.get("local_hf_path", "distilbert-base-uncased")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Build Multimodal Model
    model = MultimodalDemandEngine(config).to(device)

    # Optimizer & Scheduler
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 0.0001))
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("mixed_precision", True))
    criterion = nn.HuberLoss()

    # Dataset Files
    dataset_cfg = config.get("dataset", {})
    tfrecord_pattern = dataset_cfg.get("tfrecords_pattern", "data/processed/*.tfrecord")
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))

    if not tfrecord_files:
        raise FileNotFoundError(f"Expected .tfrecord files matching {tfrecord_pattern}, but found none!")

    print(f"✅ Found {len(tfrecord_files)} .tfrecord files! Constructing DataLoader...")

    # TFRecord Schema Description
    description = {
        "text": "byte",
        "tabular_features": "float",
        "historical_sales": "float",
        "future_demand": "float",
    }

    model.train()
    print("⚡ Starting Multimodal PyTorch Training Loop...\n")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_batches = 0

        for tf_file in tfrecord_files:
            dataset = TFRecordDataset(tf_file, index_path=None, description=description)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

            for batch in loader:
                # 1. Decode & Tokenize Text
                raw_texts = [
                    t.decode("utf-8", errors="ignore") if isinstance(t, bytes) else str(t)
                    for t in batch["text"]
                ]
                encoded = tokenizer(
                    raw_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=dataset_cfg.get("sequence_length", 128),
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                # 2. Modality Tensors
                tabular_features = batch["tabular_features"].to(device)
                historical_sales = batch["historical_sales"].to(device)
                targets = batch["future_demand"].to(device)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=train_cfg.get("mixed_precision", True)):
                    predictions = model(input_ids, attention_mask, tabular_features, historical_sales)
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
    save_dir = train_cfg.get("model_dir", "models/checkpoints/")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "multimodal_demand_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()