import torch
from torch.utils.data import Dataset


class WalmartDataset(Dataset):
    """
    Wrapper for the PyTorch Dataset class to handle the Walmart sales data.
    """

    def __init__(self, X, y):
        # Use float32 for compatibility with PyTorch's default tensor type.
        self.x = torch.tensor(X, dtype=torch.float32)
        # Add an extra dimension for the target variable.
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
