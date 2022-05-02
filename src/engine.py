import warnings

import numpy as np
import torch
from tqdm import tqdm

from .utils import compute_preds_for_classification, crop_to_bbox

warnings.filterwarnings("ignore")


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    classifier=None,
    optimizer_cls=None,
    criterion=None,
    writers=None,
):
    total_loss_dec = []

    if classifier is None:
        assert classifier is None and criterion is None and optimizer_cls is None
    else:
        classifier.train()
        total_loss_cls = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in tqdm(data_loader, desc=f"Epoch [{epoch+1}]", position=0, leave=True):
        model.train()

        # convert detection target if classification
        if classifier is not None:
            targets_dec = compute_detection_labels(targets)
            targets_cls = targets
        else:
            targets_dec = targets

        images = list(image.to(device) for image in images)
        targets_dec = [{k: v.to(device) for k, v in t.items()} for t in targets_dec]
        if classifier is not None:
            targets_cls = [{k: v.to(device) for k, v in t.items()} for t in targets_cls]

        optimizer.zero_grad()

        loss_dict = model(images, targets_dec)
        losses_dec = sum(loss for loss in loss_dict.values())

        losses_dec.backward()
        optimizer.step()

        total_loss_dec += [losses_dec.item()]

        if classifier is not None:
            # only continue if any detections
            if len([v for t in targets_cls for v in t["labels"]]) > 0:
                optimizer_cls.zero_grad()
                losses_cls, acc, labels, scores = classify_dections(
                    images, targets_cls, targets_cls, classifier, criterion, device
                )
                losses_cls.backward()
                optimizer_cls.step()
                total_loss_cls += [losses_cls.item()]

        if lr_scheduler is not None:
            lr_scheduler.step()

    if writers is not None:
        for writer in writers:
            writer.add_scalar("Loss/train/dec", np.mean(total_loss_dec), epoch)
            if classifier is not None:
                writer.add_scalar("Loss/train/cls", np.mean(total_loss_cls), epoch)

    if classifier is not None:
        return np.mean(total_loss_dec), np.mean(total_loss_cls)
    return np.mean(total_loss_dec)


@torch.inference_mode()
def validate(
    model,
    data_loader,
    device,
    metric,
    epoch,
    classifier=None,
    criterion=None,
    metric_cls=None,
    writers=None,
):
    total_loss_dec = []

    if classifier is None:
        assert classifier is None and criterion is None
    else:
        classifier.eval()
        total_loss_cls = []
        accuracy = []

    for images, targets in tqdm(data_loader, desc="Evaluating", position=0, leave=True):
        model.train()

        # convert detection target if classification
        if classifier is not None:
            targets_dec = compute_detection_labels(targets)
            targets_cls = targets
        else:
            targets_dec = targets

        images = list(image.to(device) for image in images)
        targets_dec = [{k: v.to(device) for k, v in t.items()} for t in targets_dec]
        if classifier is not None:
            targets_cls = [{k: v.to(device) for k, v in t.items()} for t in targets_cls]

        # detection loss
        loss_dict = model(images, targets_dec)
        losses_dec = sum(loss for loss in loss_dict.values())

        # detection score
        model.eval()
        with torch.no_grad():
            detections = model(images)

            metric.update(detections, targets_dec)
            total_loss_dec += [losses_dec.item()]

            # classification
            if classifier is not None:
                new_detections, new_targets = compute_preds_for_classification(detections, targets_cls)

                # only continue if any detections
                if len([v for t in new_detections for v in t["labels"]]) > 0:

                    losses_cls, acc, labels, scores = classify_dections(
                        images, new_detections, new_targets, classifier, criterion, device
                    )

                    # update detections with new labels
                    idx_label = 0
                    for detection, new_detection in zip(detections, new_detections):
                        detection["labels"] = [classifier.num_classes for _ in detection["labels"]]
                        for i_old in new_detection["index"]:
                            detection["labels"][i_old] = labels[idx_label]
                            detection["scores"][i_old] = (scores[idx_label] + detection["scores"][i_old]) / 2
                            idx_label += 1

                    detections = [{k: torch.as_tensor(v) for k, v in t.items()} for t in detections]

                    metric_cls.update(detections, targets_cls)
                    total_loss_cls += [losses_cls.item()]
                    accuracy += [acc]

    # metric
    scores_dec = metric.compute()
    metric.reset()
    if classifier is not None:
        scores_cls = metric_cls.compute()
        metric_cls.reset()

    # write
    if writers is not None:
        for writer in writers:
            writer.add_scalar("Loss/val/dec", np.mean(total_loss_dec), epoch)
            writer.add_scalars("Score/val/dec", scores_dec, epoch)
            if classifier is not None:
                writer.add_scalar("Loss/val/cls", np.mean(total_loss_dec), epoch)
                writer.add_scalars("Score/val/cls", scores_cls, epoch)
                writer.add_scalar("Acc/val/cls", np.mean(accuracy), epoch)

    # return
    if classifier is not None:
        return (
            np.mean(total_loss_dec),
            scores_dec,
            np.mean(total_loss_cls),
            scores_cls,
            np.mean(accuracy),
        )
    return np.mean(total_loss_dec), scores_dec


@torch.inference_mode()
def old_validate(model, data_loader, device, metric, epoch, writers=None):
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


def compute_detection_labels(targets):
    targets_dec = []
    for target in targets:
        target_dec = {k: v for k, v in target.items()}
        target_dec["labels"] = torch.ones(target_dec["labels"].shape, dtype=torch.int64)
        targets_dec += [target_dec]
    return targets_dec


def classify_dections(images, detections, targets, classifier, criterion, device):
    cropped_images = crop_to_bbox(images, detections, classifier.img_size)
    cropped_images = cropped_images.to(device)

    target_labels = torch.as_tensor([torch.as_tensor(v) for t in targets for v in t["labels"]]).to(device)

    outputs = classifier(cropped_images)
    labels = outputs.argmax(axis=1)
    scores = outputs.amax(axis=1)

    loss = criterion(outputs, target_labels)
    acc = np.mean((labels == target_labels).cpu().numpy())

    return loss, acc, labels, scores
