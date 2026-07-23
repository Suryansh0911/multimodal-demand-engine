import glob
import os
import torch
import torch.nn as nn
import yaml
from src.data.pipeline import build_tfrecord_dataloader
from src.models.multimodal_fusion import MultimodalDemandEngine


def main():
    # Load configuration
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

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

    # Build DataLoader from real .tfrecord files in data/processed/
    data_dir = "data/processed"
    tfrecord_files = glob.glob(f"{data_dir}/*.tfrecord")

    if not tfrecord_files:
        raise FileNotFoundError(
            f"Expected .tfrecord files in {data_dir}, but found none!"
        )

    print(
        f"✅ Found {len(tfrecord_files)} .tfrecord files! Constructing PyTorch DataLoader..."
    )

    # Option A: If tfrecord library is installed, use native parser.
    # Option B: Otherwise, stream via TensorFlow dataset casting to Torch tensors.
    import tensorflow as tf

    # Suppress TF GPU memory allocation so PyTorch owns the GPU
    tf.config.set_visible_devices([], "GPU")

    def _parse_function(proto):
        feature_description = {
            "input_ids": tf.io.FixedLenFeature([128], tf.int64),
            "attention_mask": tf.io.FixedLenFeature([128], tf.int64),
            "tabular_features": tf.io.FixedLenFeature([10], tf.float32),
            "target": tf.io.FixedLenFeature([], tf.float32),
        }
        return tf.io.parse_single_example(proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(_parse_function).batch(
        config["training"]["batch_size"]
    )

    # Training Loop
    epochs = config["training"]["epochs"]
    model.train()

    print("⚡ Starting Training Loop on Real Dataset...\n")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        step_count = 0

        for batch in parsed_dataset:
            # Cast TensorFlow tensors directly to PyTorch tensors on GPU
            input_ids = torch.tensor(batch["input_ids"].numpy(), dtype=torch.long).to(device)
            attention_mask = torch.tensor(batch["attention_mask"].numpy(), dtype=torch.long).to(device)
            tabular_features = torch.tensor(batch["tabular_features"].numpy(), dtype=torch.float32).to(device)
            targets = torch.tensor(batch["target"].numpy(), dtype=torch.float32).to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=config["training"].get("mixed_precision", True)):
                predictions = model(input_ids, attention_mask, tabular_features)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            step_count += 1

        scheduler.step()
        epoch_loss = running_loss / max(step_count, 1)
        print(f"Epoch [{epoch}/{epochs}] - Huber Loss: {epoch_loss:.4f} - RMSE: {epoch_loss**0.5:.4f}")

    # Save Checkpoint
    save_dir = config["training"].get("model_dir", "models/checkpoints/")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "multimodal_demand_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Model checkpoint successfully saved to {checkpoint_path}")


if __name__ == "__main__":
    main()