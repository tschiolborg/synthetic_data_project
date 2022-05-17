import warnings
from typing import List, Optional, Dict

import numpy as np
import torch
from tqdm import tqdm

from .utils import crop_to_bbox

warnings.filterwarnings("ignore")

__all__ = [
    "train_one_epoch_detection",
    "validate_detection",
    "evaluate_detection",
    "evalute_both_stages",
    "train_one_epoch_cls",
    "validate_cls",
]


def train_one_epoch_detection(
    model, optimizer, data_loader, device, metric, epoch: int, writers: Optional[List] = None
):
    """Train detection model for one epoch"""

    total_loss = []

    # warmup
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # iterate loader
    for images, targets in tqdm(data_loader, desc=f"Epoch [{epoch+1}]", position=0, leave=True):
        model.train()

        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # reset grad
        optimizer.zero_grad()

        # forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # backward
        losses.backward()
        optimizer.step()

        total_loss += [losses.item()]

        if lr_scheduler is not None:
            lr_scheduler.step()

        # detections to compute score
        model.eval()
        with torch.no_grad():
            detections = model(images)
            metric.update(detections, targets)

    # metric
    scores = metric.compute()
    metric.reset()

    # log
    if writers is not None:
        for writer in writers:
            writer.add_scalar("Loss/train/dec", np.mean(total_loss), epoch)
            writer.add_scalars("Score/train/dec", scores, epoch)

    return np.mean(total_loss), scores


@torch.inference_mode()
def validate_detection(
    model, data_loader, device, metric, epoch: int, writers: Optional[List] = None
):
    """Validates detection model"""

    total_loss = []

    # iterate loader
    for images, targets in tqdm(data_loader, desc="Evaluating", position=0, leave=True):

        model.train()  # for computing validation loss

        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # detection loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # detection score
        model.eval()
        with torch.no_grad():
            detections = model(images)

        # update metric
        metric.update(detections, targets)
        total_loss += [losses.item()]

    # metric
    scores = metric.compute()
    metric.reset()

    # write
    if writers is not None:
        for writer in writers:
            writer.add_scalar("Loss/val/dec", np.mean(total_loss), epoch)
            writer.add_scalars("Score/val/dec", scores, epoch)

    return np.mean(total_loss), scores


@torch.inference_mode()
def evaluate_detection(model, data_loader, device, metric):
    """Evaluates detection model on test data"""

    total_loss = []

    # iterate loader
    for images, targets in tqdm(data_loader, desc="Evaluating score", position=0, leave=True):
        model.train()  # for computing validation loss

        targets = compute_detection_labels(targets)

        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # detection loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        model.eval()

        # detections
        with torch.no_grad():
            detections = model(images)

        metric.update(detections, targets)
        total_loss += [losses.item()]

    scores = metric.compute()
    metric.reset()

    return np.mean(total_loss), scores


@torch.inference_mode()
def evalute_both_stages(
    model, classifier, data_loader, device, metric_dec, metric_cls, cls_input_size: int,
):
    """Evaluate full detection with both model + classifier on test data"""

    model.eval()
    classifier.eval()

    # iterate loader
    for images, targets in tqdm(data_loader, desc="Evaluating", position=0, leave=True):

        # transform detection targets to binary
        targets_dec = compute_detection_labels(targets)
        targets_cls = targets

        # to device
        images = list(image.to(device) for image in images)
        targets_dec = [{k: v.to(device) for k, v in t.items()} for t in targets_dec]
        targets_cls = [{k: v.to(device) for k, v in t.items()} for t in targets_cls]

        # detection
        with torch.no_grad():
            detections = model(images)

        # update metric with detection only
        metric_dec.update(detections, targets_dec)

        # only continue if any detections
        if len([v for t in detections for v in t["labels"]]) > 0:

            # get cropped detections
            cropped_images = crop_to_bbox(images, detections, cls_input_size)
            cropped_images = cropped_images.to(device)

            # classification
            outputs = classifier(cropped_images).cpu()
            labels = outputs.argmax(axis=1)
            scores = torch.nn.functional.softmax(outputs, dim=1).amax(axis=1)

            # copy detections
            updated_detections = [{k: v.cpu().clone() for k, v in t.items()} for t in detections]

            # update detections with new labels
            i = 0
            for detections_on_one_img in updated_detections:
                if len(detections_on_one_img["labels"]) == 0:
                    continue

                for j in range(len(detections_on_one_img["labels"])):
                    detections_on_one_img["labels"][j] = labels[i + j]
                    detections_on_one_img["scores"][j] += scores[i + j]
                    detections_on_one_img["scores"][j] /= 2
                i += j + 1

            # update metric with classification
            metric_cls.update(updated_detections, targets_cls)

            # print()
            # a = 0
            # for d2, t in zip(updated_detections, targets_cls):
            #     if len(d2["labels"]) == 0:
            #         continue

            #     print(t)

            #     for k in range(len(d2["labels"])):
            #         print(d2["labels"][k], d2["scores"][k], d2["boxes"][k])

            #         plt.imshow(cropped_images.cpu()[a + k].permute(1, 2, 0))
            #         plt.show()

            #     a += k + 1

            #     print()
            #     print()
            #     print()

    # compute metric
    scores_dec = metric_dec.compute()
    metric_dec.reset()
    scores_cls = metric_cls.compute()
    metric_cls.reset()

    return scores_dec, scores_cls


def train_one_epoch_cls(
    classifier,
    optimizer,
    criterion,
    data_loader,
    device,
    epoch: int,
    cls_input_size: int,
    writers: Optional[List] = None,
):
    """Train classification model for one epoch"""

    classifier.train()
    total_loss = []
    accuracy = []

    # iterate loader
    for images, targets in tqdm(data_loader, desc=f"Epoch [{epoch+1}]", position=0, leave=True):

        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # do classification
        optimizer.zero_grad()
        losses, acc, labels, scores = classify_dections(
            images=images,
            detections=targets,  # detection = targets for using real bboxes
            targets=targets,
            classifier=classifier,
            criterion=criterion,
            device=device,
            img_size=cls_input_size,
        )

        # backward
        losses.backward()
        optimizer.step()

        total_loss += [losses.item()]
        accuracy += [acc]

    # log
    if writers is not None:
        for writer in writers:
            writer.add_scalar("Loss/train/cls", np.mean(total_loss), epoch)
            writer.add_scalar("Acc/train/cls", np.mean(accuracy), epoch)

    return np.mean(total_loss), np.mean(accuracy)


@torch.inference_mode()
def validate_cls(
    classifier, criterion, data_loader, device, epoch: int, cls_input_size: int, writers=None,
):
    """Validation of classification model"""

    classifier.eval()
    total_loss_cls = []
    accuracy = []

    # iterate loader
    for images, targets in tqdm(data_loader, desc="Evaluating", position=0, leave=True):

        # to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # do classfication
        with torch.no_grad():
            losses, acc, labels, scores = classify_dections(
                images=images,
                detections=targets,  # detection = targets for using real bboxes
                targets=targets,
                classifier=classifier,
                criterion=criterion,
                device=device,
                img_size=cls_input_size,
            )

        total_loss_cls += [losses.item()]
        accuracy += [acc]

    # log
    if writers is not None:
        for writer in writers:
            writer.add_scalar("Loss/val/cls", np.mean(total_loss_cls), epoch)
            writer.add_scalar("Acc/val/cls", np.mean(accuracy), epoch)

    return (np.mean(total_loss_cls), np.mean(accuracy))


def compute_detection_labels(targets: List[Dict]):
    """Returns targets with binary labels"""

    targets_dec = []
    for target in targets:
        target_dec = {k: v for k, v in target.items()}
        target_dec["labels"] = torch.ones(target_dec["labels"].shape, dtype=torch.int64)
        targets_dec += [target_dec]
    return targets_dec


def classify_dections(images, detections, targets, classifier, criterion, device, img_size: int):
    """
    Performs classification on boxes on images
    targets and detections must be of same size, such that each detection corresponds to a target
    img_size: the input size for the classifier
    """

    # list of images for all detections
    cropped_images = crop_to_bbox(images, detections, img_size)

    # to device
    cropped_images = cropped_images.to(device)
    target_labels = torch.as_tensor([torch.as_tensor(v) for t in targets for v in t["labels"]]).to(
        device
    )

    # classify
    outputs = classifier(cropped_images)
    labels = outputs.argmax(axis=1)
    scores = outputs.amax(axis=1)

    # loss and metric
    loss = criterion(outputs, target_labels)
    acc = np.mean((labels == target_labels).cpu().numpy())

    return loss, acc, labels, scores


def update_detections_with_cls_labels(
    old_detections: List[Dict],
    new_detections: List[Dict],
    labels: List[int],
    scores: List[float],
    num_classes: int,
):
    """Applies new labels to detections based on their label from classification"""

    # copy detections
    old_detections_copy = [{k: v.cpu().clone() for k, v in t.items()} for t in old_detections]

    # assign new labels
    idx_label = 0
    for detection, new_detection in zip(old_detections_copy, new_detections):
        detection["labels"] = [num_classes for _ in detection["labels"]]
        for i_old in new_detection["index"]:
            detection["labels"][i_old] = labels[idx_label]
            detection["scores"][i_old] = (scores[idx_label] + detection["scores"][i_old]) / 2
            idx_label += 1

    return [{k: torch.as_tensor(v) for k, v in t.items()} for t in old_detections_copy]
