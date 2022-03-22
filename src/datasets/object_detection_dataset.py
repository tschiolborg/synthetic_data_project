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
from src.transforms import get_transform, safe_transform
from src.utils.load_files import load_image


class ObjectDetectionDataset(Dataset):
    def __init__(
        self,
        image_ids: List,
        image_dir: str,
        anno_dir: str,
        transforms: bool,  # Optional[Compose] = None,
        classes: Optional[Dict[str, int]] = None,
    ):
        """
        Dataset for object detection.
        Annotations should be contained in a single directory
        with one annotation json file for each image.
        Each file should contain [objects], that each contain [label] : string and [bbox]
        each [bbox] should contain [xmin], [ymin], [xmax] and [ymax] : float

        Args:
            image_ids: list of ids (name+extension) to images
            images_path: path to images
            anno_dir: path to annotation directory
            classes: dictionary mapping classes to integer label (if None then all labels)
            transforms: albumentations
            mode: train/val/test
        """
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.classes = classes
        self.transforms = transforms
        self.num_classes = len(classes) if classes is not None else -1  # -1 for all classes

    def __len__(self):
        return len(self.image_ids)

    def _load_target(self, id):
        anno_dir = os.path.join(self.anno_dir, f"{Path(id).stem}.json")

        boxes = []
        labels = []

        with open(anno_dir) as f:
            anno = json.load(f)

            for obj in anno["objects"]:
                label = obj["label"]
                box = obj["bbox"]
                if self.classes is None or label in self.classes:
                    labels.append(self.classes[label]["id"])
                    boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torchvision.ops.box_area(boxes)
        img_key = id

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "img_key": img_key,
        }
        return target

    def __getitem__(self, idx):
        id = self.image_ids[idx]
        image = load_image(Path(id).stem, self.image_dir)
        target = self._load_target(id)

        if self.transforms is not None:

            transforms = get_transform(self.transforms, height=image.shape[0], width=image.shape[1])

            sample = transforms(image=image, bboxes=target["boxes"], labels=target["labels"])

            max_tries = 100
            while len(sample["bboxes"]) == 0:
                # retry until the bbox is acceptable
                # self.log.info("Retrying target transforms.")
                sample = transforms(image=image, bboxes=target["boxes"], labels=target["labels"])

                max_tries -= 1
                if max_tries <= 0:
                    transforms = safe_transform(self.transforms)

            image = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])
            target["area"] = torchvision.ops.box_area(target["boxes"])

        else:
            image = torch.as_tensor(image, dtype=torch.float32)

        return image, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))
