import torch
import torch.nn as nn
from src.models.text_encoder import TextEncoder


class MultimodalDemandEngine(nn.Module):

    def __init__(self, config):
        super().__init__()
        text_backbone = config.get("model", {}).get(
            "text_backbone", "distilbert-base-uncased"
        )
        self.text_encoder = TextEncoder(
            model_name=text_backbone, embed_dim=128
        )

        # Update input dimension from 10 to 15 (matching your actual dataset)
        num_tabular_features = config.get("model", {}).get(
            "num_tabular_features", 15
        )

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
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, tabular_features):
        text_emb = self.text_encoder(input_ids, attention_mask)
        tab_emb = self.tabular_encoder(tabular_features)

        # Concatenate embeddings
        fused = torch.cat([text_emb, tab_emb], dim=1)
        output = self.regressor(fused)
        return output.squeeze(-1)