# src/models/multimodal_fusion.py

import torch
import torch.nn as nn
from src.models.text_encoder import TextEncoder


class MultimodalDemandEngine(nn.Module):

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})

        # 1. Text Encoder (DistilBERT)
        local_hf_path = model_cfg.get(
            "local_hf_path", "distilbert-base-uncased"
        )
        text_dim = model_cfg.get("text_embedding_dim", 128)
        self.text_encoder = TextEncoder(
            model_name=local_hf_path, embed_dim=text_dim
        )

        # 2. Tabular Encoder
        num_tabular = model_cfg.get("num_tabular_features", 15)
        tabular_dim = model_cfg.get("tabular_embedding_dim", 64)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular, tabular_dim),
            nn.ReLU(),
            nn.BatchNorm1d(tabular_dim),
            nn.Linear(tabular_dim, tabular_dim),
            nn.ReLU(),
        )

        # 3. Temporal / Historical Sales Encoder (1D CNN)
        temporal_dim = model_cfg.get("temporal_embedding_dim", 128)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, temporal_dim),
            nn.ReLU(),
        )

        # 4. Fusion Head (128 text + 64 tabular + 128 temporal = 320 features)
        total_fusion_dim = text_dim + tabular_dim + temporal_dim
        self.regressor = nn.Sequential(
            nn.Linear(total_fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, self_or_input_ids, attention_mask=None, tabular_features=None, historical_sales=None
    ):  # Ensure all 4 input tensors are present in signature!
        # If positional args match standard signature:
        if attention_mask is None:
            # Safety fallback for single arg call
            return self_or_input_ids

        input_ids = self_or_input_ids

        # Embed text
        text_emb = self.text_encoder(input_ids, attention_mask)

        # Embed tabular
        tab_emb = self.tabular_encoder(tabular_features)

        # Embed historical sales sequence: shape (batch_size, 1, sequence_length)
        if historical_sales.dim() == 2:
            historical_sales = historical_sales.unsqueeze(1)
        temp_emb = self.temporal_encoder(historical_sales)

        # Concatenate all 3 modalities
        fused = torch.cat([text_emb, tab_emb, temp_emb], dim=1)
        output = self.regressor(fused)
        return output.squeeze(-1)