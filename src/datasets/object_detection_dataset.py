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
        image_dir: str,
        anno_dir: str,
        classes: Optional[Dict[str, int]] = None,
        transforms: Optional[Compose] = None,
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
        self.transform = transforms
        self.num_classes = len(classes) if classes is not None else -1  # -1 for all classes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_ids[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255

        anno_dir = os.path.join(self.anno_dir, f"{Path(self.image_ids[idx]).stem}.json")

        boxes = []
        labels = []

        with open(anno_dir) as f:
            anno = json.load(f)

            for obj in anno["objects"]:
                label = obj["label"]
                box = obj["bbox"]
                if self.classes is None or label in self.classes:
                    labels.append(self.classes[label])
                    boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torchvision.ops.box_area(boxes)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "image_id": torch.tensor([idx]),
        }

        if self.transform is not None:

            sample = self.transform(image=image, bboxes=target["boxes"], labels=labels)

            while len(sample["bboxes"]) == 0:
                # retry until the bbox is acceptable
                # self.log.info("Retrying target transforms.")
                sample = self.transform(image=image, bboxes=target["boxes"], labels=labels)

            image = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])
            target["area"] = torchvision.ops.box_area(target["boxes"])

        else:
            image = torch.as_tensor(image, dtype=torch.float32)

        return image, target

    # @staticmethod
    # def collate_fn(batch):
    #    return tuple(zip(*batch))
