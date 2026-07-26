import torch
from torch import nn


class SalesPredictor(nn.Module):
    def __init__(self, cont_dim, num_stores=46, num_depts=100, emb_dim=10):
        super().__init__()

        # Embeddings encode information about the individual stores and
        # departments. Better than simply storing them as numbers in
        # increasing order, as that can be misleading information for a
        # neural network.
        self.store_emb = nn.Embedding(num_stores, emb_dim)
        self.dept_emb = nn.Embedding(num_depts, emb_dim)

        # Calculate the input dimension
        input_dim = cont_dim + (emb_dim * 2)

        self.network = nn.Sequential(
            # Hidden layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),  # 20% dropout rate to prevent overfitting
            # Hidden layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Hidden layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Output layer
            nn.Linear(64, 1),
        )

    def forward(self, x_cont, x_cat):
        # Extract embeddings
        store_embeds = self.store_emb(x_cat[:, 0])
        dept_embeds = self.dept_emb(x_cat[:, 1])

        # Concatenate the continuous features with the embeddings
        x = torch.cat([x_cont, store_embeds, dept_embeds], dim=1)

        return self.network(x)
