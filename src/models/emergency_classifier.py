import torch
import torch.nn as nn

class MimiEmergencyClassifier(nn.Module):
    def __init__(self, num_codebooks=32, codebook_size=2048, embed_dim=128):
        super().__init__()

        # Embedding for each codebook token
        self.embedding = nn.Embedding(codebook_size, embed_dim)

        # Combine codebooks into one representation per frame
        self.codebook_proj = nn.Linear(num_codebooks * embed_dim, embed_dim)

        # Temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, codes):
        # codes: (B, 32, T)
        B, C, T = codes.shape

        x = self.embedding(codes)          # (B, 32, T, embed_dim)
        x = x.permute(0, 2, 1, 3)          # (B, T, 32, embed_dim)
        x = x.reshape(B, T, -1)            # (B, T, 32*embed_dim)

        x = self.codebook_proj(x)          # (B, T, embed_dim)
        x = self.temporal_encoder(x)       # (B, T, embed_dim)

        x = x.mean(dim=1)                  # Temporal pooling

        logits = self.classifier(x).squeeze(-1)
        return torch.sigmoid(logits)

