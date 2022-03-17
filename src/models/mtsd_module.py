from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from omegaconf import DictConfig


class MtsdLitModule(LightningModule):
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

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # metric
        self.val_metric = MeanAveragePrecision()

    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch: Any, batch_idx: int):
        images, targets = batch
        # targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.forward(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # log train loss
        self.log("train_loss", losses, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": losses, "log": loss_dict, "progress_bar": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        images, targets = batch
        # images = list(img for img in images)
        # targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.forward(images, targets)

        # update val metrics
        self.val_metric.update(outputs, targets)

        return {}

    def validation_epoch_end(self, outputs: List[Any]):
        mAP = self.val_metric.compute()
        self.log("val_mAP", mAP, on_epoch=True, prog_bar=False)
        log = {"main_score": mAP}

        print(mAP)

        self.val_metric.reset()  # reset metric at the end of every epoch
        return {"val_mAP": mAP, "log": log, "progress_bar": log}

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(params, lr=self.cfg.train.lr, momentum=0.9, weight_decay=0.0005)
