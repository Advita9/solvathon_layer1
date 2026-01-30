import torch

def mimi_collate(batch):
    codes_list, labels = zip(*batch)

    max_len = max(c.shape[1] for c in codes_list)
    padded_codes = []

    for c in codes_list:
        pad_len = max_len - c.shape[1]
        if pad_len > 0:
            pad = torch.zeros(c.shape[0], pad_len, dtype=torch.long)
            c = torch.cat([c, pad], dim=1)
        padded_codes.append(c)

    padded_codes = torch.stack(padded_codes)
    labels = torch.tensor(labels)

    return padded_codes, labels
