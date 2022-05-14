import os

import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50

from src.config import Config
from src.engine import train_one_epoch_cls, validate_cls
from src.utils import (
    Json_writer,
    collate_fn,
    load_data,
    load_optimizer,
)


def train_cls(cfg: Config):
    """Train classification model"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    print(f"Directory: {os.getcwd()}")

    num_classes = cfg.training.num_classes

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
        classifier = torch.load(
            os.path.join(cfg.checkpoint.path, cfg.utils.model_dir, "classifier.pkl")
        )
    else:
        classifier = resnet50(pretrained=True)
        classifier.fc = torch.nn.Linear(2048, num_classes)

    # set up epochs
    start_epoch = 0
    num_epochs = cfg.training.epochs

    # optimizer + criterion
    params_cls = [p for p in classifier.parameters() if p.requires_grad]
    optimizer_cls = load_optimizer(cfg, params_cls)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    accuracies = []

    # set up logging
    log_dir_json = os.path.join(os.getcwd(), cfg.utils.log_dir_json)
    os.makedirs(log_dir_json, exist_ok=True)
    writers = [
        Json_writer(log_file=os.path.join(log_dir_json, "run.json")),
        # SummaryWriter(log_dir=os.path.join(os.getcwd(), cfg.utils.log_dir)),
    ]

    # resume at checkpoint
    if cfg.checkpoint.resume:
        checkpoint_cls = torch.load(
            os.path.join(cfg.checkpoint.path, cfg.utils.model_dir, "ckpt_cls.pth")
        )
        classifier.load_state_dict(checkpoint_cls["classifier_state_dict"])
        optimizer_cls.load_state_dict(checkpoint_cls["optimizer_cls_state_dict"])
        start_epoch = checkpoint_cls["epoch"] + 1
        writers[0].load_data(os.path.join(cfg.checkpoint.path, cfg.utils.log_dir_json, "run.json"))

    # to device
    classifier = classifier.to(device)

    # training loop
    for epoch in range(start_epoch, num_epochs):

        # train
        loss, train_acc = train_one_epoch_cls(
            classifier=classifier,
            optimizer=optimizer_cls,
            criterion=criterion,
            data_loader=data_loader_train,
            device=device,
            epoch=epoch,
            cls_input_size=224,
            writers=writers,
        )

        # validate
        loss, acc = validate_cls(
            classifier=classifier,
            criterion=criterion,
            data_loader=data_loader_val,
            device=device,
            epoch=epoch,
            cls_input_size=224,
            writers=writers,
        )

        accuracies += [acc]

    # log
    for writer in writers:
        writer.flush()

    # save
    model_dir = os.path.join(os.getcwd(), cfg.utils.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(classifier, f"{model_dir}/classifier.pkl")
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_cls_state_dict": optimizer_cls.state_dict(),
            "epoch": epoch,
        },
        f"{model_dir}/ckpt_cls.pth",
    )

    print(accuracies[-1])
    print(f"\nSaved at: {os.getcwd()}")

    return


@hydra.main(config_path="conf/", config_name="config_cls.yaml")
def run_cls(cfg: Config) -> None:
    train_cls(cfg)


if __name__ == "__main__":
    run_cls()
