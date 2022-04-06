import warnings

import numpy as np
import torch
import torchvision
from tqdm import tqdm

warnings.filterwarnings("ignore")


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, writers=None):
    model.train()

    total_loss = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch [{epoch+1}]", position=0, leave=True)):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

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

        total_loss += [losses.item()]

    for writer in writers:
        if writer is not None:
            writer.add_scalar("Loss/train", np.mean(total_loss), epoch)

    return np.mean(total_loss)


@torch.inference_mode()
def validate(model, data_loader, device, metric, epoch, writers=None):
    model.eval()

    for images, targets in tqdm(data_loader, desc="Evaluating score", position=0, leave=True):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=False):
            detections = model(images, targets)

        metric.update(detections, targets)

    scores = metric.compute()
    metric.reset()


    model.train() # to get loss

    total_loss = []

    for images, targets in tqdm(data_loader, desc="Evaluating loss", position=0, leave=True):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += [losses.item()]

    for writer in writers:
        if writer is not None:
            writer.add_scalar("Loss/val", np.mean(total_loss), epoch)
            writer.add_scalars("Score/val", scores, epoch)

    return np.mean(total_loss), scores


@torch.inference_mode()
def evaluate(model, data_loader, device, metric):
    model.eval()

    for images, targets in tqdm(data_loader, desc="Evaluating score", position=0, leave=True):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=False):
            detections = model(images, targets)

        metric.update(detections, targets)

    scores = metric.compute()
    metric.reset()

    return scores


@torch.inference_mode()
def predict(img, model, device):
    model.eval()

    pred = model([img.to(device)])[0]

    keep = torchvision.ops.nms(pred['boxes'] , pred['scores'], 0.1)
    return pred, keep
