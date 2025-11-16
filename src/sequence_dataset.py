import torch
from torch.utils.data import Dataset

class SessionDataset(Dataset):
    def __init__(self, df, feature_cols, target_col="skip", max_len=50):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.max_len = max_len

        # Group by session
        self.sessions = []
        grouped = df.groupby("session_id")

        for session_id, g in grouped:
            g = g.sort_values("position")
            X = g[feature_cols].values
            y = g[target_col].values
            self.sessions.append((X, y))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        X, y = self.sessions[idx]

        # pad
        pad_len = self.max_len - len(X)
        if pad_len > 0:
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            X_pad = torch.zeros(pad_len, len(self.feature_cols))
            y_pad = torch.zeros(pad_len)
            X = torch.cat([X, X_pad], dim=0)
            y = torch.cat([y, y_pad], dim=0)
        else:
            X = torch.tensor(X[:self.max_len], dtype=torch.float32)
            y = torch.tensor(y[:self.max_len], dtype=torch.long)

        return X, y


def collate_fn(batch):
    xs = []
    ys = []
    lengths = []

    for X, y in batch:
        lengths.append(len(y))
        xs.append(torch.tensor(X))
        ys.append(torch.tensor(y))

    xs = torch.stack(xs)
    ys = torch.stack(ys)

    return xs, ys, torch.tensor(lengths)
