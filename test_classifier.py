import os

import hydra
import torch

from src.config import ConfigTest
from src.engine import validate_cls
from src.utils import collate_fn, load_data_test


def test_cls():
    with hydra.initialize(config_path="conf"):
        cfg: ConfigTest = hydra.compose(config_name="config_test_cls.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    print(f"Directory: {os.getcwd()}")

    dataset_test = load_data_test(cfg)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.dataset.test.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn,
    )

    model = torch.load(cfg.model_dir)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    loss, acc = validate_cls(model, criterion, data_loader_test, device=device)

    print(f"Loss: {loss}")
    print(f"Accuray: {acc}")

    return acc


if __name__ == "__main__":
    test_cls()
