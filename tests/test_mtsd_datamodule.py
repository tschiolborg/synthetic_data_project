import os

import pytest
import torch
import hydra

from src.datamodules.mtsd_datamodule import MtsdDataModule
from src.utils.load_files import ROOT


@pytest.mark.parametrize("batch_size", [1])
def test_mtsd_datamodule(batch_size):

    conf_dir = os.path.join(ROOT, "conf")

    assert os.path.exists(conf_dir)

    with hydra.initialize_config_dir(config_dir=conf_dir):
        cfg = hydra.compose(config_name="testing.yaml")
        cfg.datamodule.train.batch_size = batch_size

    datamodule = MtsdDataModule(cfg=cfg)
    datamodule.prepare_data()

    assert (
        not datamodule.train_dataset and not datamodule.val_dataset and not datamodule.test_dataset
    )

    assert os.path.exists(os.path.join("data", "MTSD"))
    assert os.path.exists(os.path.join("data", "MTSD", "images", "train"))
    assert os.path.exists(os.path.join("data", "MTSD", "images", "val"))
    assert os.path.exists(os.path.join("data", "MTSD", "images", "test"))
    assert os.path.exists(os.path.join("data", "MTSD", "annotations"))
    assert os.path.exists(os.path.join("data", "classes"))

    datamodule.setup()

    assert datamodule.train_dataset and datamodule.val_dataset and datamodule.test_dataset
    assert (
        len(datamodule.train_dataset) > 0
        and len(datamodule.val_dataset) > 0
        and len(datamodule.test_dataset) > 0
    )

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    images, targets = next(iter(datamodule.train_dataloader()))

    assert len(images) == batch_size
    assert len(targets) == batch_size

    image = images[0]
    target = targets[0]

    assert image.dtype == torch.float32

    assert target.get("labels") and target.get("boxes") and target.get("area")
    assert target["labels"].dtype == torch.int64
    assert target["boxes"].dtype == torch.float32
    assert target["area"].dtype == torch.float32
