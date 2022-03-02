import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from albumentations.core.composition import Compose


class MtsdDataset(Dataset):
    def __init__(
        self,
        image_ids: List,
        img_path: str,
        anno_path: str,
        classes: Optional[Dict[str, int]] = None,
        transforms: Optional[Compose] = None,
        mode: str = "",
    ):
        """
        Initialize the (lazy-loader) dataloader.
        Args:
            image_ids: list of ids to images
            img_path: path to images
            anno_path: path to annotation directory
            classes: dictionary mapping classes to integer label
            transforms: albumentations
            mode: train/val/test
        """
        self.image_ids = image_ids
        self.img_path = img_path
        self.anno_path = anno_path
        self.classes = classes
        self.transform = transforms
        self.mode = mode
        self.num_classes = len(classes) if classes is not None else -1 # -1 for all classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_ids[idx]}{self.extension}")
        img_bgr = cv2.imread(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255

        anno_path = os.path.join(self.anno_path, f"{self.__get_annotation_id_(idx)}.json")

        boxes = []
        labels = []

        with open(anno_path) as f:
            anno = json.load(f)

            for _obj in anno["objects"]:
                cls = _obj["label"]
                box = _obj["bbox"]
                if self.__should_load_anno_(cls):
                    labels.append(self.classes.index(cls))
                    boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)  # must be of int64

        area = torchvision.ops.box_area(boxes)

        targets = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "image_id": torch.tensor([idx]),
        }

        if self.transform:

            sample = self.transform(image=img, bboxes=targets["boxes"], labels=labels)

            while len(sample["bboxes"]) == 0:
                # retry until the bbox is acceptable
                self.log.info("Retrying target transforms.")
                sample = self.transform(image=img, bboxes=targets["boxes"], labels=labels)

            img = sample["image"]
            targets["boxes"] = torch.Tensor(sample["bboxes"])
            targets["area"] = torchvision.ops.box_area(targets["boxes"])

        else:
            img = torch.as_tensor(img, dtype=torch.float32)

        return img, targets

    def __should_load_anno_(self, cls: str) -> bool:
        if self.classes is None:
            return True
        elif cls in self.classes:
            return True

        return False

    def __get_annotation_id_(self, idx: int):
        if self.cfg.use_single_annotation_file:
            return "annotations"
        return self.img_ids[idx]

    def __filter_ids_for_training_(self, ids: List[str]) -> List[str]:
        """
        This function checks which image examples contain the classes we desire to detect.
        This function should only be used when the annotations are stored in multiple files.
        """
        filtered_ids: List[str] = []

        for _id in ids:
            anno_path = os.path.join(self.anno_path, f"{_id}.json")
            with open(anno_path) as f:
                anno = json.load(f)
                for _obj in anno["objects"]:
                    if _obj["label"] in self.classes:
                        filtered_ids.append(_id)
                        break  # We use break to avoid adding duplicate ids
        return filtered_ids

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
