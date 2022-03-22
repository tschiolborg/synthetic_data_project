import os

import pytest
import torch
import hydra

from src.datamodules.mtsd_datamodule import MtsdDataModule
from src.utils.load_files import ROOT, load_classes


@pytest.mark.parametrize("batch_size", [1, 4])
def test_mtsd_datamodule(batch_size):

    conf_dir = os.path.join(ROOT, "conf")

    assert os.path.exists(conf_dir)

    with hydra.initialize_config_dir(config_dir=conf_dir):
        cfg = hydra.compose(config_name="debug.yaml")
        cfg.datamodule.train.batch_size = batch_size

    # classes
    num_classes = cfg.datamodule.num_classes
    classes = load_classes(cfg.datamodule.classes_path)

    assert len(classes) == num_classes
    for label in classes:
        assert classes[label]["id"] < num_classes and classes[label]["id"] >= 0

    # datamodule
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

    assert datamodule.train_dataset and datamodule.val_dataset
    assert len(datamodule.train_dataset) > 0 and len(datamodule.val_dataset) > 0

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()

    images, targets = next(iter(datamodule.train_dataloader()))

    assert len(images) == batch_size
    assert len(targets) == batch_size

    image = images[0]
    target = targets[0]

    assert image.dtype == torch.float32

    assert (
        target.get("labels") is not None
        and target.get("boxes") is not None
        and target.get("area") is not None
    )
    assert target["labels"].dtype == torch.int64
    assert target["boxes"].dtype == torch.float32
    assert target["area"].dtype == torch.float32

    # target
    for target in targets:
        labels = target["labels"].numpy()
        boxes = target["boxes"]

        for label in labels:
            assert label < num_classes and label >= 1

        for box in boxes:
            assert box[0] < box[2] and box[1] < box[3]
