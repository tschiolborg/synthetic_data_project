import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torchvision.transforms import ToTensor


class MTSD_Dataset(torch.utils.data.Dataset):
    """Dataset for Mappilary Traffic Sign Dataset"""

    def __init__(
        self,
        image_dir,
        anno_dir,
        extension="jpg",
        transforms=None,
        only_detect=False,
        threshold=900,
        keep_other=False,
    ):
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.extension = extension
        self.transforms = transforms
        self.only_detect = only_detect
        self.threshold = threshold
        self.keep_other = keep_other
        self.ids = sorted([Path(anno).stem for anno in os.listdir(anno_dir)])

    def get_no_transform(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.ids[idx]}.{self.extension}")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255

        anno_path = os.path.join(self.anno_dir, f"{self.ids[idx]}.json")
        with open(anno_path) as f:
            anno = json.load(f)

        target = {"labels": [], "labels_str": [], "boxes": [], "areas": []}

        for label, label_str, box, area in zip(
            anno["labels"], anno["labels_str"], anno["boxes"], anno["areas"]
        ):
            if area >= self.threshold and (label_str != "other-sign" or self.keep_other):
                target["labels"].append(label)
                target["labels_str"].append(label_str)
                target["boxes"].append(box)
                target["areas"].append(area)

        return img, target

    def __getitem__(self, idx):
        img, _target = self.get_no_transform(idx)

        target = {}

        target["boxes"] = torch.as_tensor(_target["boxes"], dtype=torch.float32)

        target["labels"] = (
            torch.as_tensor(_target["labels"], dtype=torch.int64)
            if not self.only_detect
            else torch.ones(len(_target["labels"]), dtype=torch.int64)
        )

        local_transforms = self.transforms(height=img.shape[0], width=img.shape[1])

        if self.transforms is not None:
            sample = local_transforms(image=img, bboxes=target["boxes"], labels=target["labels"])

            max_tries = 100
            while len(sample["bboxes"]) == 0:
                sample = local_transforms(
                    image=img, bboxes=target["boxes"], labels=target["labels"]
                )

                max_tries -= 1
                if max_tries <= 0:
                    break

            img = sample["image"]
            target["boxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(sample["labels"], dtype=torch.int64)
        else:
            img = ToTensorV2()(image=img)["image"]

        if len(target["boxes"]) == 0:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0), dtype=torch.int64)

        target["image_id"] = torch.as_tensor(idx)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


class GTSDB_Dataset(torch.utils.data.Dataset):
    """Dataset for the German Traffic Sign Detection Benchmark"""

    def __init__(self, image_dir, anno_dir, transforms=None, only_detect=False, mtsd_labels=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.only_detect = only_detect
        self.mtsd_labels = mtsd_labels
        self.imgs = list(sorted(os.listdir(image_dir)))[:300]

        anno_file = np.genfromtxt(
            os.path.join(anno_dir, "gt.txt"), delimiter=";", dtype=None, encoding=None
        )

        if self.mtsd_labels is not None:
            with open(self.mtsd_labels) as f:
                classes_map = json.load(f)
                classes_map = {int(k): v for (k, v) in classes_map.items()}

        anno_dict = {}

        for img_id, xmin, ymin, xmax, ymax, label in anno_file:
            if self.mtsd_labels is not None:
                label = classes_map[label]
            if label is None:
                continue

            if img_id in anno_dict:
                anno_dict[img_id]["boxes"] += [[xmin, ymin, xmax, ymax]]
                anno_dict[img_id]["labels"] += [label]
            else:
                anno_dict[img_id] = {}
                anno_dict[img_id]["boxes"] = [[xmin, ymin, xmax, ymax]]
                anno_dict[img_id]["labels"] = [label]

        self.anno_dict = anno_dict

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        target = self.anno_dict.get(self.imgs[idx])

        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)

            target["labels"] = (
                torch.as_tensor(target["labels"], dtype=torch.int64)
                if not self.only_detect
                else torch.ones(len(target["labels"]), dtype=torch.int64)
            )

        else:
            target = {}
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0), dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


class SYNTH_Dataset(torch.utils.data.Dataset):
    """Dataset for the Synthetic Traffic Sign Dataset"""

    def __init__(
        self, image_dir, anno_dir, train: bool, transforms=None, only_detect: bool = False
    ):
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        self.only_detect = only_detect

        # split into train or test
        imgs = list(sorted(os.listdir(image_dir)))
        split_idx = int(len(imgs) * 0.8)
        self.imgs = imgs[:split_idx] if train else imgs[split_idx:]
        annos = list(sorted(os.listdir(anno_dir)))
        self.annos = annos[:split_idx] if train else annos[split_idx:]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        target_path = os.path.join(self.anno_dir, self.annos[idx])
        with open(target_path) as f:
            target = json.load(f)

        if len(target["boxes"]) == 0:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0), dtype=torch.int64)
        else:
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["areas"] = torch.as_tensor(target["areas"], dtype=torch.float32)

            target["labels"] = (
                torch.as_tensor(target["labels"], dtype=torch.int64)
                if not self.only_detect
                else torch.ones(len(target["labels"]), dtype=torch.int64)
            )

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
