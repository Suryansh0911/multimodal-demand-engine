import torch
from torch.utils.data import DataLoader, Dataset


class DemandDataset(Dataset):

    def __init__(self, data_list):
        """data_list: list of dicts containing processed inputs"""
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                item["attention_mask"], dtype=torch.long
            ),
            "tabular_features": torch.tensor(
                item["tabular_features"], dtype=torch.float32
            ),
            "target": torch.tensor(item["target"], dtype=torch.float32),
        }


def build_dataloader(dataset, batch_size=32, is_training=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=2,
        pin_memory=True,
    )