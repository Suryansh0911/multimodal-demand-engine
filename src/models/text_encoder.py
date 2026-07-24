import os
import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):

    def __init__(self, model_name="distilbert-base-uncased", embed_dim=128):
        super().__init__()

        if not os.path.exists(model_name):
            model_name = "distilbert-base-uncased"

        self.transformer = AutoModel.from_pretrained(model_name)

        for param in self.transformer.parameters():
            param.requires_grad = False

        if hasattr(self.transformer, "transformer") and hasattr(
            self.transformer.transformer, "layer"
        ):
            for param in self.transformer.transformer.layer[
                -1
            ].parameters():
                param.requires_grad = True

        hidden_size = self.transformer.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embed_dim), nn.ReLU(), nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_output)