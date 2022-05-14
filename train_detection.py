import os

import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn

from src.config import Config
from src.engine import train_one_epoch_detection, validate_detection
from src.utils import (
    Json_writer,
    collate_fn,
    load_data,
    load_lr_scheduler,
    load_optimizer,
)


def train_detection(cfg: Config):
    """Trains detection model"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    print(f"Directory: {os.getcwd()}")

    num_classes = 2 if cfg.training.only_detect else cfg.training.num_classes

    # load data
    dataset_train, dataset_val = load_data(cfg)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.dataset.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.dataset.val.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn,
    )

    # load model
    if cfg.checkpoint.resume:
        model = torch.load(os.path.join(cfg.checkpoint.path, cfg.utils.model_dir, "model.pkl"))
    else:
        if cfg.training.use_coco:
            model = fasterrcnn_resnet50_fpn(pretrained=True, max_size=4000)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        else:
            model = fasterrcnn_resnet50_fpn(
                pretrained=False, num_classes=num_classes, pretrained_backbone=True, max_size=4000
            )

    # optimizer and lr scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = load_optimizer(cfg, params)
    lr_scheduler = load_lr_scheduler(cfg, optimizer)

    # set up epochs
    start_epoch = 0
    num_epochs = cfg.training.epochs

    # metric
    val_metric = MeanAveragePrecision()
    val_scores = []

    # set up loggers
    log_dir_json = os.path.join(os.getcwd(), cfg.utils.log_dir_json)
    os.makedirs(log_dir_json, exist_ok=True)
    writers = [
        Json_writer(log_file=os.path.join(log_dir_json, "run.json")),
        # SummaryWriter(log_dir=os.path.join(os.getcwd(), cfg.utils.log_dir)),
    ]

    # resume at checkpoint
    if cfg.checkpoint.resume:
        checkpoint = torch.load(os.path.join(cfg.checkpoint.path, cfg.utils.model_dir, "ckpt.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        writers[0].load_data(os.path.join(cfg.checkpoint.path, cfg.utils.log_dir_json, "run.json"))

    # do device
    model = model.to(device)

    # training loop
    for epoch in range(start_epoch, num_epochs):

        # train model
        loss = train_one_epoch_detection(
            model, optimizer, data_loader_train, device=device, epoch=epoch, writers=writers
        )

        lr_scheduler.step()

        # validate
        loss, scores = validate_detection(
            model, data_loader_val, device=device, metric=val_metric, epoch=epoch, writers=writers,
        )

        val_scores += [scores]

    # log
    for writer in writers:
        writer.flush()

    # save
    model_dir = os.path.join(os.getcwd(), cfg.utils.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, f"{model_dir}/model.pkl")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        },
        f"{model_dir}/ckpt.pth",
    )

    print(val_scores[-1])
    print(f"\nSaved at: {os.getcwd()}")

    return


@hydra.main(config_path="conf/", config_name="config.yaml")
def run_model(cfg: Config) -> None:
    train_detection(cfg)


if __name__ == "__main__":
    run_model()
