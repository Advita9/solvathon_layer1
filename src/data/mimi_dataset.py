import json
import torch
from torch.utils.data import Dataset

class MimiEmergencyDataset(Dataset):
    def __init__(self, split):
        with open("data/splits.json") as f:
            files = json.load(f)[split]

        self.samples = []
        for path in files:
            label = 1 if "emergency" in path else 0
            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path)
        return data["acoustic_codes"].long(), torch.tensor(label, dtype=torch.float32)
