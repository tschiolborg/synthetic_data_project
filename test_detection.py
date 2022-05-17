import os

import hydra
import torch
from torchmetrics.detection.map import MeanAveragePrecision

from src.config import ConfigTest
from src.engine import evaluate_detection
from src.utils import collate_fn, load_data_test


def test_detection():
    """Test detection model on test data"""

    with hydra.initialize(config_path="conf"):
        cfg: ConfigTest = hydra.compose(config_name="config_test.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # load data
    dataset_test = load_data_test(cfg)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.dataset.test.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn,
    )

    # load model
    model = torch.load(cfg.model_dir)

    metric = MeanAveragePrecision()

    model = model.to(device)

    # evaluate model
    loss, score = evaluate_detection(
        model=model, data_loader=data_loader_test, device=device, metric=metric
    )

    print(score)
    print(loss)

    return loss, score


if __name__ == "__main__":
    test_detection()
