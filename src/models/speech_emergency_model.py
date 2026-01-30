import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class SpeechEmergencyEmbeddingModel(nn.Module):
    def __init__(self, base_model="facebook/wav2vec2-base", embed_dim=256):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out = self.encoder(x).last_hidden_state
        pooled = out.mean(dim=1)
        embedding = self.projection(pooled)
        logits = self.classifier(embedding).squeeze(-1)
        return embedding, logits

