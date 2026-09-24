"""Dataset wrapper, model forward pass, and the committed checkpoint."""

import torch
from torch.utils.data import DataLoader

from src.preprocessor import DataPreprocessor
from src.sales_predictor import SalesPredictor
from src.walmart_dataset import WalmartDataset
from tests.conftest import REPO_ROOT

CHECKPOINT = REPO_ROOT / "saved_weights" / "best_model.pth"


def _dataset(frame):
    pre = DataPreprocessor()
    return pre, WalmartDataset(*pre.fit_transform(frame))


def test_dataset_items_have_training_dtypes(train_frame):
    _, ds = _dataset(train_frame)
    x_cont, x_cat, y = ds[0]
    assert len(ds) == len(train_frame)
    assert x_cont.dtype == torch.float32
    assert x_cat.dtype == torch.long
    assert y.shape == (1,)


def test_forward_pass_shape(train_frame):
    pre, ds = _dataset(train_frame)
    model = SalesPredictor(cont_dim=len(pre.cont_cols))
    x_cont, x_cat, _ = next(iter(DataLoader(ds, batch_size=8)))
    assert model(x_cont, x_cat).shape == (8, 1)


def test_a_few_steps_reduce_training_loss(train_frame):
    torch.manual_seed(0)
    pre, ds = _dataset(train_frame)
    x_cont, x_cat, y = ds.x_cont, ds.x_cat, ds.y
    model = SalesPredictor(cont_dim=len(pre.cont_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        initial = loss_fn(model(x_cont, x_cat), y).item()

    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss_fn(model(x_cont, x_cat), y).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final = loss_fn(model(x_cont, x_cat), y).item()
    assert final < initial / 2


def test_committed_checkpoint_matches_model_definition():
    # Guards against changing the architecture without retraining the
    # weights that the README and notebook rely on.
    state = torch.load(CHECKPOINT, weights_only=True)
    model = SalesPredictor(cont_dim=len(DataPreprocessor().cont_cols))
    model.load_state_dict(state, strict=True)
