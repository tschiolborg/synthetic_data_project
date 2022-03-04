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


class ObjectDetectionDataset(Dataset):
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
        Dataset for object detection.
        Annotations should be contained in a single directory
        with one annotation json file for each image.
        Each file should contain [objects], that each contain [label] : string and [bbox]
        each [bbox] should contain [xmin], [ymin], [xmax] and [ymax] : float

        Args:
            image_ids: list of ids (name+extension) to images
            images_path: path to images
            anno_path: path to annotation directory
            classes: dictionary mapping classes to integer label (if None then all labels)
            transforms: albumentations
            mode: train/val/test
        """
        self.image_ids = image_ids
        self.img_path = img_path
        self.anno_path = anno_path
        self.classes = classes
        self.transform = transforms
        self.mode = mode
        self.num_classes = len(classes) if classes is not None else -1  # -1 for all classes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_path, self.img_ids[idx])
        image = cv2.imread(image_path)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255

        anno_path = os.path.join(self.anno_path, f"{Path(self.image_ids[idx]).stem}.json")

        boxes = []
        labels = []

        with open(anno_path) as f:
            anno = json.load(f)

            for obj in anno["objects"]:
                label = obj["label"]
                box = obj["bbox"]
                if self.__should_load_anno_(label):
                    labels.append(self.classes.index(label))
                    boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

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

    # @staticmethod
    # def collate_fn(batch):
    #    return tuple(zip(*batch))
