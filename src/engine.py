import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, classifier=None, criterion=None, writers=None,
):
    if classifier is None or criterion is None:
        assert classifier is None and criterion is None
    else:
        classifier.train()

    model.train()

    loss_total = []
    loss_dec = []
    loss_cls = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in tqdm(data_loader, desc=f"Epoch [{epoch+1}]", position=0, leave=True):
        model.train()

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses_dec = sum(loss for loss in loss_dict.values())

        if classifier is not None:
            model.eval()
            with torch.no_grad():
                detections = model(images)
            detections = T.Resize(classifier.img_size)(detections)
            output = classifier(detections)
            losses_cls = criterion(output, targets)  ## fix this

        else:
            losses_cls = 0

        losses = losses_dec + losses_cls

        losses.backward()
        optimizer.step()

        loss_total += [losses.item()]

        if classifier is not None:
            loss_dec += [losses_dec.item()]
            loss_cls += [losses_cls.item()]

        if lr_scheduler is not None:
            lr_scheduler.step()

    if writers is not None:
        for writer in writers:
            writer.add_scalar("Loss/train", np.mean(loss_total), epoch)
            if classifier is not None:
                writer.add_scalar("Loss/train/dec", np.mean(loss_dec), epoch)
                writer.add_scalar("Loss/train/cls", np.mean(loss_cls), epoch)

    return np.mean(loss_total)


@torch.inference_mode()
def validate(model, data_loader, device, metric, epoch, writers=None):
    model.eval()

    for images, targets in tqdm(data_loader, desc="Evaluating score", position=0, leave=True):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        detections = model(images)

        metric.update(detections, targets)

    scores = metric.compute()
    metric.reset()

    model.train()  # to get loss

    total_loss = []

    for images, targets in tqdm(data_loader, desc="Evaluating loss", position=0, leave=True):
        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += [losses.item()]

    if writers is not None:
        for writer in writers:
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

    keep = torchvision.ops.nms(pred["boxes"], pred["scores"], 0.1)
    return pred, keep
