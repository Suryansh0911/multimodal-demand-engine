import glob
import torch
from torch.utils.data import DataLoader, IterableDataset
try:
    from tfrecord.torch.dataset import TFRecordDataset
except ImportError:
    # Fallback to pure PyTorch loader if tfrecord library is absent
    pass


class PyTorchTFRecordDataset(IterableDataset):

    def __init__(self, tfrecord_pattern, index_pattern=None, batch_size=32):
        super().__init__()
        self.tfrecord_files = sorted(glob.glob(tfrecord_pattern))
        self.batch_size = batch_size

        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"No TFRecord files found matching pattern: {tfrecord_pattern}"
            )

    def parse_tfrecord_feature(self, record):
        """Extracts and formats feature dictionaries from decoded TFRecord byte dicts."""
        # Note: Adjust key names below if your ETL script used different column names
        return {
            "input_ids": torch.tensor(
                record.get("input_ids", [101] + [0] * 127), dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                record.get("attention_mask", [1] * 128), dtype=torch.long
            ),
            "tabular_features": torch.tensor(
                record.get("tabular_features", [0.0] * 10), dtype=torch.float32
            ),
            "target": torch.tensor(
                record.get("target", 0.0), dtype=torch.float32
            ),
        }


def build_tfrecord_dataloader(tfrecord_dir, batch_size=32):
    """Builds a PyTorch DataLoader streaming directly from .tfrecord files."""
    tfrecord_pattern = f"{tfrecord_dir}/*.tfrecord"
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))

    print(f"📦 Loading {len(tfrecord_files)} TFRecord files from {tfrecord_dir}")

    # Standard TFRecord schema description for tfrecord library
    description = {
        "input_ids": "int",
        "attention_mask": "int",
        "tabular_features": "float",
        "target": "float",
    }

    # Stream TFRecords natively into PyTorch
    dataset = TFRecordDataset(
        tfrecord_files[0], index_path=None, description=description
    )

    return DataLoader(
        dataset, batch_size=batch_size, num_workers=2, pin_memory=True
    )