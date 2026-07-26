import torch
from torch.utils.data import Dataset


class WalmartDataset(Dataset):
    """
    Wrapper for the PyTorch Dataset class to handle the Walmart sales data.
    """

    def __init__(self, x_cont, x_cat, y):
        # Use float32 for compatibility with PyTorch's default tensor type.
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        # Use torch.long for embedding indices
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        return self.x_cont[idx], self.x_cat[idx], self.y[idx]
