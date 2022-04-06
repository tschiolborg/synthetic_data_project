from typing import Any, List

import torch
import torchvision
from pytorch_lightning import LightningModule

from torchmetrics.detection.map import MeanAveragePrecision

# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from omegaconf import DictConfig

from torchvision.models.detection.rpn import AnchorGenerator

from ..imported.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, FasterRCNN


class FasterRCNNLitningModule(LightningModule):
    """
    PyTorch Lightning module for object detection
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)

        self.num_classes = cfg.datamodule.num_classes

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0, 4.0, 8.0, 16.0),)
        )

        self.model = fasterrcnn_resnet50_fpn(pretrained=True, rpn_anchor_generator=anchor_generator)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # backbone = torchvision.models.resnet50(pretrained=True)

        # self.model = FasterRCNN(backbone=backbone)

        # metric
        self.val_metric = MeanAveragePrecision()

    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch: Any, batch_idx: int):
        images, targets = batch

        loss_dict, _ = self.forward(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # log train loss
        self.log("train/loss", losses, on_step=False, on_epoch=True, prog_bar=False)

        return {"train/loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        images, targets = batch

        loss_dict, detections = self.forward(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        self.log("val/loss", losses, on_step=False, on_epoch=True, prog_bar=False)

        # update val metrics
        self.val_metric.update(detections, targets)

        return {"val/loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    def validation_epoch_end(self, outputs: List[Any]):
        metric = self.val_metric.compute()
        self.log("val/AP", metric, on_epoch=True, prog_bar=False)
        log = {"val/metric": metric}

        print(metric)

        self.val_metric.reset()  # reset metric at the end of every epoch
        return {"val/metric": metric, "log": log, "progress_bar": log}

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(params, lr=self.cfg.train.lr, momentum=0.9, weight_decay=0.0005)
