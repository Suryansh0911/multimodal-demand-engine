import glob
import os
import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer
from tfrecord.torch.dataset import TFRecordDataset
from src.models.multimodal_fusion import MultimodalDemandEngine


def main():
    # 1. Load configuration
    config_path = "configs/train_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # 2. Tokenizer setup
    model_cfg = config.get("model", {})
    hf_path = model_cfg.get("local_hf_path", "distilbert-base-uncased")

    if not os.path.exists(hf_path):
        hf_path = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(hf_path)

    # 3. Model setup
    model = MultimodalDemandEngine(config).to(device)

    # 4. Optimizer, Scheduler & Loss Setup
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 0.0001))
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("mixed_precision", True))
    criterion = nn.HuberLoss()

    # 5. Dataset Setup
    dataset_cfg = config.get("dataset", {})
    tfrecord_pattern = dataset_cfg.get("tfrecords_pattern", "data/processed/*.tfrecord")
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))

    if not tfrecord_files:
        raise FileNotFoundError(f"Expected .tfrecord files matching '{tfrecord_pattern}', but found none!")

    print(f"✅ Found {len(tfrecord_files)} .tfrecord files! Setting up DataLoader...")

    description = {
        "text": "byte",
        "tabular_features": "float",
        "historical_sales": "float",
        "future_demand": "float",
    }

    model.train()
    print("⚡ Starting Multimodal Training Loop...\n")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_batches = 0

        for tf_file in tfrecord_files:
            dataset = TFRecordDataset(tf_file, index_path=None, description=description)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)

            for batch in loader:
                # A. Decode & Tokenize Text
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

                # B. Extract Modality Tensors safely
                tabular_features = batch["tabular_features"].to(device).float()
                historical_sales = batch["historical_sales"].to(device).float()

                # C. Target shape normalization to 1D: [batch_size]
                targets = batch["future_demand"].to(device).float().reshape(-1)

                optimizer.zero_grad()

                # D. Forward & Backward Pass with AMP
                with torch.amp.autocast("cuda", enabled=train_cfg.get("mixed_precision", True)):
                    predictions = model(input_ids, attention_mask, tabular_features, historical_sales)
                    # Force predictions to 1D: [batch_size] to prevent loss broadcasting
                    predictions = predictions.reshape(-1)
                    loss = criterion(predictions, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                total_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(total_batches, 1)
        print(f"Epoch [{epoch}/{epochs}] - Huber Loss: {epoch_loss:.4f} - RMSE: {epoch_loss**0.5:.4f}")

    # 6. Save Checkpoint
    save_dir = train_cfg.get("model_dir", "models/checkpoints/")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "multimodal_demand_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Model checkpoint successfully saved to {checkpoint_path}")


if __name__ == "__main__":
    main()