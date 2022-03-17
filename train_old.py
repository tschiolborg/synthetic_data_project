import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import hydra
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from HydraNet.conf.config import DatasetConfig, PipelineConfig
from src.transforms import get_transform
from src.utils.visualize import insert_box, show_image

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


@hydra.main(config_path="HydraNet/conf", config_name="config.yaml")
def train(cfg: PipelineConfig):
    """
    Train
    """

    # log.info("Running main.py")
    # print(cfg.train.path)

    cfg_model = cfg.model
    cfg_dataset = cfg.train.dataset
    cfg_dataloder = cfg.train.dataloader
    cfg_dataset_val = cfg.train.dataset_val
    cfg_dataloder_val = cfg.train.dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    num_classes = cfg.model.num_classes

    # use dataset with transformations
    dataset = MtsdDataset(cfg=cfg_dataset, transform=get_transform(train=True))
    dataset_val = MtsdDataset(cfg=cfg_dataset_val, transform=get_transform(train=False))

    print(f"Size of training data: {len(dataset)}")
    print(f"Size of validation data: {len(dataset_val)}")

    # define data loaders
    # TODO: create class wrt config
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=MtsdDataset.collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=MtsdDataset.collate_fn,
    )

    # load model pre-trained on coco
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 5

    val_metric = MeanAveragePrecision()

    for epoch in range(num_epochs):
        # train for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device, val_metric=val_metric)

    print("Done")

    with torch.no_grad():
        model.eval()
        model = model.to(device)

        images, targets = next(iter(data_loader_test))
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        preds = model(images, targets)
        print(targets)
        print(preds)

        for img, target, pred in zip(images, targets, preds):
            img = img.cpu().permute((1, 2, 0))
            for box in target["boxes"]:
                img = insert_box(img, box.cpu(), "1")
            for box in pred["boxes"]:
                img = insert_box(img, box.cpu(), "2")
            show_image(img)

    # model_name = os.getcwd().split("\\")[-1] + ".pth"
    # print(model_name)
    # torch.save(model.state_dict(), model_name)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200, scaler=None):
    """
    Train one epoch
    dont know if scaler works
    """
    model.train()
    model = model.to(device)

    total_loss = 0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for idx, (images, targets) in enumerate(
        tqdm(data_loader, desc=f"Epoch [{epoch+1}]", position=0, leave=True)
    ):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += losses

        # change to log
        # if idx % print_freq == 0:
        #    print(f"Loss: {losses}")

    print(f"Total loss: {total_loss}")


@torch.inference_mode()
def evaluate(model, data_loader, device, val_metric):
    """
    Evalue
    """
    model.eval()

    for images, targets in tqdm(data_loader, desc="Evaluating", position=0, leave=True):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        val_metric.update(outputs, targets)

    metric = val_metric.compute()
    print(metric)
    val_metric.reset()


#
#
#
#
#
#
#
#
#
#


class MtsdDataset(Dataset):
    """
    Dataset that can load from one or multiple annotation files. It is possible to specify which classes to load.
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        extension: str = ".jpg",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the (lazy-loader) dataloader.
        Args:
            cfg.img_path: string path to image folder.
            cfg.anno_path: string path to annotation folder.
            cfg.classes: list of strings with class labels that should be loaded. `None` will load all classes.
            cfg.use_single_annotation_file: indicates whether annotations are stored in one or multiple files.
            cfg.label_map: dictionary mapping from class label to int.
            extension: image file extension.
            transform: optional callable that transforms the input images.
            target_transform: optional callable that transforms the input annotations.
        """
        self.cfg = cfg
        self.anno_dir = cfg.anno_dir
        self.img_dir = cfg.img_dir
        self.classes = cfg.classes
        self.label_map = cfg.label_map
        self.extension = extension
        self.img_ids = [
            Path(file).stem
            for file in os.listdir(self.img_dir)
            if Path(file).suffix == self.extension
        ]
        if not self.cfg.use_single_annotation_file:
            self.img_ids = self.__filter_ids_for_training_(self.img_ids)

        self.transform = transform
        self.log = logging.getLogger(__name__)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_ids[idx]}{self.extension}")
        img_bgr = cv2.imread(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255

        anno_path = os.path.join(self.anno_dir, f"{self.__get_annotation_id_(idx)}.json")

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
            anno_path = os.path.join(self.anno_dir, f"{_id}.json")
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


if __name__ == "__main__":
    train()
