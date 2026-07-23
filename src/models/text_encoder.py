import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):

    def __init__(self, model_name="distilbert-base-uncased", embed_dim=128):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        # Freeze Hugging Face backbone parameters
        for param in self.transformer.parameters():
            param.requires_grad = False

        hidden_size = self.transformer.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embed_dim), nn.ReLU(), nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Use [CLS] token / first token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_output)