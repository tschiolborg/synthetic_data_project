import logging
import math
import sys

import hydra
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from dataset import MtsdDataset
from HydraNet.conf.config import PipelineConfig
from src.transforms import get_transform

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
    cfg_dataset_val = cfg.train.dataset
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

    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

    print("Done")


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200, scaler=None):
    """
    Train one epoch
    dont know if scaler works
    """
    model.train()
    model = model.to(device)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch [{epoch+1}]")):
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

        # change to log
        if idx % print_freq == 0:
            print(f"Loss: {losses}")


@torch.inference_mode()
def evaluate(model, data_loader, device):
    """
    Evalue
    """
    model.eval()

    for idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # TODO


if __name__ == "__main__":
    train()
