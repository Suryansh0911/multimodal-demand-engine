import torch
import torch.nn as nn
from src.models.text_encoder import TextEncoder


class MultimodalDemandEngine(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(
            model_name=config["model"].get(
                "text_backbone", "distilbert-base-uncased"
            ),
            embed_dim=128,
        )

        # Tabular / Temporal Feature Extractor
        num_tabular_features = config["model"].get("num_tabular_features", 10)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion Head (128 text + 64 tabular = 192 features)
        fusion_dim = 128 + 64
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Regression output: predicted demand/sales
        )

    def forward(self, input_ids, attention_mask, tabular_features):
        text_emb = self.text_encoder(input_ids, attention_mask)
        tab_emb = self.tabular_encoder(tabular_features)

        # Concatenate embeddings along feature dimension
        fused = torch.cat([text_emb, tab_emb], dim=1)
        output = self.regressor(fused)
        return output.squeeze(-1)