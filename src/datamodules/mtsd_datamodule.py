from typing import Optional
import os
from pathlib import Path
import json

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datasets import ObjectDetectionDataset as MyDataset
from omegaconf import DictConfig


class MtsdDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        with open(os.path.join(self.cfg.datamodule.classes_path)) as f:
            self.classes = json.load(f)

        self.num_classes = len(self.classes)

        self.train_dataset = (
            self._setup_dataset(self.cfg.datamodule.train) if self.cfg.datamodule.train else None
        )

        self.val_dataset = (
            self._setup_dataset(self.cfg.datamodule.val) if self.cfg.datamodule.val else None
        )

        self.test_dataset = (
            self._setup_dataset(self.cfg.datamodule.test) if self.cfg.datamodule.test else None
        )

    def train_dataloader(self):
        return (
            DataLoader(
                dataset=self.train_dataset,
                batch_size=self.cfg.datamodule.train.batch_size,
                num_workers=self.cfg.datamodule.train.num_workers,
                pin_memory=self.cfg.datamodule.train.pin_memory,
                shuffle=True,
                collate_fn=None,
                drop_last=False,
            )
            if self.train_dataset is not None
            else None
        )

    def val_dataloader(self):
        return (
            DataLoader(
                dataset=self.val_dataset,
                batch_size=self.cfg.datamodule.val.batch_size,
                num_workers=self.cfg.datamodule.val.num_workers,
                pin_memory=self.cfg.datamodule.val.pin_memory,
                shuffle=False,
                collate_fn=None,
                drop_last=False,
            )
            if self.val_dataset is not None
            else None
        )

    def test_dataloader(self):
        return (
            DataLoader(
                dataset=self.test_dataset,
                batch_size=self.cfg.datamodule.test.batch_size,
                num_workers=self.cfg.datamodule.test.num_workers,
                pin_memory=self.cfg.datamodule.test.pin_memory,
                shuffle=False,
                collate_fn=None,
                drop_last=False,
            )
            if self.test_dataset is not None
            else None
        )

    def _setup_dataset(self, cfg_dataset):
        images = [file for file in os.listdir(cfg_dataset.path)]

        if not self.cfg.datamodule.include_negative_examples:
            images = [id for id in images if self._filter_id(id)]

        if self.cfg.training.debug:
            images = images[:1000]

        return MyDataset(
            image_ids=images,
            img_path=cfg_dataset.path,
            anno_path=self.cfg.datamodule.anno_path,
            classes=self.classes,
            transforms=cfg_dataset.transforms,
        )

    def _filter_id(self, id):
        anno_path = os.path.join(self.cfg.datamodule.anno_path, f"{Path(id).stem}.json")
        with open(anno_path) as f:
            anno = json.load(f)
            for obj in anno["objects"]:
                if obj["label"] in self.classes:
                    return True
        return False
