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

        with open(os.path.join(self.cfg.datamodule.classes_dir)) as f:
            self.classes = json.load(f)

        self.num_classes = len(self.classes)

        train_images = [file for file in os.listdir(self.cfg.datamodule.train.path)]
        val_images = [file for file in os.listdir(self.cfg.datamodule.val.path)]
        test_images = [file for file in os.listdir(self.cfg.datamodule.test.path)]

        if not self.cfg.datamodule.include_negative_examples:
            for images in (train_images, val_images, test_images):
                images = [id for id in images if self._filter_id(id)]

        if self.cfg.training.debug:
            for images in (train_images, val_images, test_images):
                images = images[:1000]

        self.train_dataset = MyDataset(
            image_ids=train_images,
            img_path=self.cfg.datamodule.train.path,
            anno_path=self.cfg.datamodule.anno_path,
            classes=self.classes,
            transforms=self.cfg.datamodule.train.transforms,
            mode="train",
        )

        self.val_dataset = MyDataset(
            image_ids=val_images,
            img_path=self.cfg.datamodule.val.path,
            anno_path=self.cfg.datamodule.anno_path,
            classes=self.classes,
            transforms=self.cfg.datamodule.val.transforms,
            mode="val",
        )

        self.test_dataset = MyDataset(
            image_ids=test_images,
            img_path=self.cfg.datamodule.test.path,
            anno_path=self.cfg.datamodule.anno_path,
            classes=self.classes,
            transforms=self.cfg.datamodule.test.transforms,
            mode="test",
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.datamodule.train.batch_size,
            num_workers=self.cfg.datamodule.train.num_workers,
            pin_memory=self.cfg.datamodule.train.pin_memory,
            shuffle=True,
            collate_fn=None,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.datamodule.train.batch_size,
            num_workers=self.cfg.datamodule.train.num_workers,
            pin_memory=self.cfg.datamodule.train.pin_memory,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.datamodule.train.batch_size,
            num_workers=self.cfg.datamodule.train.num_workers,
            pin_memory=self.cfg.datamodule.train.pin_memory,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )

    def _filter_id(self, id):
        anno_path = os.path.join(self.cfg.datamodule.anno_path, f"{Path(id).stem}.json")
        with open(anno_path) as f:
            anno = json.load(f)
            for obj in anno["objects"]:
                if obj["label"] in self.classes:
                    return True
        return False
