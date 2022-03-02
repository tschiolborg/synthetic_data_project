from typing import Optional
import os
from pathlib import Path
import json

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datasets import MtsdDataset
from omegaconf import DictConfig

__all__ = ["MtsdDataModule"]


class MtsdDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # this line allows to access init params with 'self.hparams' attribute
        # self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        with open(os.path.join(self.cfg.datamodule.classes_dir)) as f:
            self.classes = json.load(f)

        self.num_classes = len(self.classes)

        train_images = [Path(file).stem for file in os.listdir(self.cfg.datamodule.train.path)]
        val_images = [Path(file).stem for file in os.listdir(self.cfg.datamodule.val.path)]
        test_images = [Path(file).stem for file in os.listdir(self.cfg.datamodule.test.path)]

        if not self.cfg.datamodule.include_negative_examples:
            for images in (train_images, val_images, test_images):
                images = self.__filter_ids_for_training_(images)

        if self.cfg.training.debug:
            for images in (train_images, val_images, test_images):
                images = images[:1000]

        self.train_dataset = MtsdDataset(
            image_ids=train_images,
            img_path=self.cfg.datamodule.train.path,
            anno_path=self.cfg.datamodule.anno_path,
            classes=self.classes,
            transforms=self.cfg.datamodule.train.transforms,
            mode="train",
        )

        self.val_dataset = MtsdDataset(
            image_ids=val_images,
            img_path=self.cfg.datamodule.val.path,
            img_path=self.cfg.datamodule.train.path,
            anno_path=self.cfg.datamodule.anno_path,
            classes=self.classes,
            transforms=self.cfg.datamodule.val.transforms,
            mode="val",
        )

        self.test_dataset = MtsdDataset(
            image_ids=test_images,
            img_path=self.cfg.datamodule.test.path,
            img_path=self.cfg.datamodule.train.path,
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

    def __filter_ids_for_training_(self, ids):
        """
        This function checks which image examples contain the classes we desire to detect.
        This function should only be used when the annotations are stored in multiple files.
        """
        filtered_ids = []

        for _id in ids:
            anno_path = os.path.join(self.cfg.datamodule.anno_path, f"{_id}.json")
            with open(anno_path) as f:
                anno = json.load(f)
                for _obj in anno["objects"]:
                    if _obj["label"] in self.classes:
                        filtered_ids.append(_id)
                        break  # We use break to avoid adding duplicate ids
        return filtered_ids
