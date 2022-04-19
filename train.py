import os

import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn

from src.config import Config
from src.engine import train_one_epoch, validate
from src.utils import Json_writer, collate_fn, load_data, load_lr_scheduler, load_optimizer


def train(cfg: Config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    print(f"Directory: {os.getcwd()}")

    only_detect = cfg.training.only_detect
    num_classes = 2 if only_detect else cfg.training.num_classes

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

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = load_optimizer(cfg, params)
    lr_scheduler = load_lr_scheduler(cfg, optimizer)
    start_epoch = 0

    num_epochs = cfg.training.epochs

    val_metric = MeanAveragePrecision()

    train_losses = []
    val_losses = []
    val_scores = []

    writers = [
        SummaryWriter(log_dir=os.path.join(os.getcwd(), cfg.utils.log_dir)),
        Json_writer(log_file=os.path.join(os.getcwd(), cfg.utils.log_dir, "run.json")),
    ]

    if cfg.checkpoint.resume:
        checkpoint = torch.load(os.path.join(cfg.checkpoint.path, cfg.utils.model_dir, "ckpt.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        writers[1].load_data(os.path.join(cfg.checkpoint.path, cfg.utils.log_dir, "run.json"))

    model = model.to(device)

    for epoch in range(start_epoch, num_epochs):
        loss = train_one_epoch(
            model, optimizer, data_loader_train, device=device, epoch=epoch, writers=writers
        )
        train_losses += [loss]

        lr_scheduler.step()

        loss, scores = validate(
            model, data_loader_val, device=device, metric=val_metric, epoch=epoch, writers=writers
        )
        val_losses += [loss]
        val_scores += [scores]

    for writer in writers:
        writer.flush()

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

    return train_losses, val_losses, val_scores


@hydra.main(config_path="conf/", config_name="config.yaml")
def run_model(cfg: Config) -> None:
    train(cfg)


if __name__ == "__main__":
    run_model()
