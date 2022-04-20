import json
from msilib.schema import Class
import os

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import ImageColor, ImageDraw, ImageFont

from .config import Config, ConfigTest
from .datasets import GTSDB_Dataset, MTSD_Dataset
from .transforms import Transforms
from .classifier import SimpleNet, Classifier

dotenv.load_dotenv(override=True)


def load_data(cfg: Config):
    """
    Load and return training and validation datasets

    cfg: config file, see config.py
    cfg.dataset.name: either "MTSD" or "GTSDB"
    """
    my_transforms = Transforms(
        min_area_train=cfg.dataset.train.transforms.min_area,
        img_size_train=cfg.dataset.train.transforms.img_size,
        min_area_val=cfg.dataset.val.transforms.min_area,
        img_size_val=cfg.dataset.val.transforms.img_size,
    )

    if cfg.dataset.name == "MTSD":

        MTSD = os.getenv("MTSD")
        if not MTSD:
            raise Exception('Not able to find "MTSD" environment variable')

        img_dir = os.path.join(MTSD, "images")
        anno_train = os.path.join(MTSD, "anno_train")
        anno_val = os.path.join(MTSD, "anno_val")

        dataset_train = MTSD_Dataset(
            img_dir,
            anno_train,
            transforms=my_transforms.get_transform(True),
            only_detect=cfg.training.only_detect,
            threshold=cfg.dataset.threshold,
            keep_other=cfg.dataset.keep_other,
        )
        dataset_val = MTSD_Dataset(
            img_dir,
            anno_val,
            transforms=my_transforms.get_transform(False),
            only_detect=cfg.training.only_detect,
            threshold=cfg.dataset.threshold,
            keep_other=cfg.dataset.keep_other,
        )

    elif cfg.dataset.name == "GTSDB":

        GTSDB = os.getenv("GTSDB")
        if not GTSDB:
            raise Exception('Not able to find "GTSDB" environment variable')

        dataset_train = GTSDB_Dataset(
            os.path.join(GTSDB, "train"),
            GTSDB,
            transforms=my_transforms.get_transform_gtsdb(True),
            only_detect=cfg.training.only_detect,
            mtsd_labels=cfg.dataset.mtsd_labels,
        )
        dataset_val = GTSDB_Dataset(
            os.path.join(GTSDB, "test"),
            GTSDB,
            transforms=my_transforms.get_transform_gtsdb(False),
            only_detect=cfg.training.only_detect,
            mtsd_labels=cfg.dataset.mtsd_labels,
        )
    else:
        raise Exception(f"error cannot find dataset: {cfg.dataset.name}")

    return dataset_train, dataset_val


def load_data_test(cfg: ConfigTest):
    """
    Load and return test dataset

    cfg: config file, see config.py
    cfg.dataset.name: either "MTSD" or "GTSDB"
    """
    my_transforms = Transforms(
        min_area_val=cfg.dataset.test.transforms.min_area, img_size_val=cfg.dataset.test.transforms.img_size,
    )

    if cfg.dataset.name == "MTSD":

        MTSD = os.getenv("MTSD")
        if not MTSD:
            raise Exception('Not able to find "MTSD" environment variable')

        img_dir = os.path.join(MTSD, "images")
        anno_test = os.path.join(MTSD, "anno_val")

        dataset_test = MTSD_Dataset(
            img_dir, anno_test, transforms=my_transforms.get_transform(False), only_detect=cfg.testing.only_detect,
        )

    elif cfg.dataset.name == "GTSDB":

        GTSDB = os.getenv("GTSDB")
        if not GTSDB:
            raise Exception('Not able to find "GTSDB" environment variable')

        dataset_test = GTSDB_Dataset(
            os.path.join(GTSDB, "images"),
            GTSDB,
            transforms=my_transforms.get_transform_gtsdb(False),
            only_detect=cfg.testing.only_detect,
            mtsd_labels=cfg.dataset.mtsd_labels,
        )
    else:
        raise Exception(f"error cannot find dataset: {cfg.dataset.name}")

    return dataset_test


def load_optimizer(cfg: Config, params):
    """
    Loads learning rate scheduler. 
    Must be one of: SGD, Adam
    """
    if cfg.optimizer.name == "SGD":
        return torch.optim.SGD(
            params,
            lr=cfg.optimizer.params["lr"],
            momentum=cfg.optimizer.params["momentum"],
            weight_decay=cfg.optimizer.params["weight_decay"],
        )
    elif cfg.optimizer.name == "Adam":
        return torch.optim.Adam(params, lr=cfg.optimizer.params["lr"], eps=cfg.optimizer.params["eps"])
    else:
        raise Exception(f"error cannot find optimizer: {cfg.optimizer.name}")


def load_lr_scheduler(cfg: Config, optimizer):
    """
    Loads learning rate scheduler. 
    Must be one of: StepLR
    """
    if cfg.lr_scheduler.name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_scheduler.params["step_size"], gamma=cfg.lr_scheduler.params["gamma"],
        )
    else:
        raise Exception(f"error cannot find optimizer: {cfg.optimizer.name}")


def load_classifier(cfg: Config):
    """
    Loads classifier
    Must be one of: SimpleNet, Classifier
    """
    if cfg.classifier.name == "SimpleNet":
        return SimpleNet(cfg.training.num_classes, 28)
    elif cfg.classifier.name == "Classifier":
        return Classifier(cfg.training.num_classes, 40)
    else:
        raise Exception(f"error cannot find net: {cfg.classifier.name}")


def load_criterion(cfg: Config):
    """
    Loads criterion
    Must be one of: CrossEntropyLoss
    """
    if cfg.classifier.criterion == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    else:
        raise Exception(f"error cannot find net: {cfg.classifier.name}")


class Json_writer:
    """
    For logging training in json file 
    """

    def __init__(self, log_file):
        self.data = {}
        self.log_file = log_file

    def add_scalar(self, tag, value, epoch):
        if torch.is_tensor(value):
            value = value.item()
        if tag in self.data:
            self.data[tag] += [value]
        else:
            self.data[tag] = [value]

    def add_scalars(self, tag, value_dict, epoch):
        if tag in self.data:
            for key, value in value_dict.items():
                if torch.is_tensor(value):
                    value = value.item()
                self.data[tag][key] += [value]
        else:
            self.data[tag] = {}
            for key, value in value_dict.items():
                if torch.is_tensor(value):
                    value = value.item()
                self.data[tag][key] = [value]

    def flush(self):
        with open(self.log_file, "w+") as f:
            json.dump(self.data, f, indent=6)

    def load_data(self, log_file):
        with open(log_file) as f:
            self.data = json.load(f)
        return self


def collate_fn(batch):
    return tuple(zip(*batch))


def plot_loss(train_losses, val_losses=None, file_out="output"):
    """
    matplotlib figure of losses
    """
    epochs = range(len(train_losses))
    fig = plt.figure(figsize=(10, 8))
    plt.plot(epochs, train_losses, label="train loss")
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epochs", fontdict={"size": 12})
    plt.ylabel("Loss", fontdict={"size": 12})
    plt.title("Loss", fontdict={"size": 16})
    plt.legend(prop={"size": 12})
    plt.savefig(file_out)
    plt.show()


def plot_val(scores, labels, y_label="mAP", file_out="output"):
    """
    matplotlib figure of validation scores
    """
    epochs = range(len(scores[0]))
    fig = plt.figure(figsize=(10, 8))
    for score, label in zip(scores, labels):
        plt.plot(epochs, score, label=label)
    plt.xlabel("Epochs", fontdict={"size": 12})
    plt.ylabel(y_label, fontdict={"size": 12})
    plt.title("Averge Precision", fontdict={"size": 16})
    plt.legend(prop={"size": 12})
    plt.savefig(file_out)
    plt.show()


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str=""):
    """
    draws a single box and displays label on PIL image
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color,
    )

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_height = (1 + 2 * 0.05) * font.getsize(display_str)[1]

    if top > display_str_height:
        text_bottom = top
    else:
        text_bottom = top + display_str_height

    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color,
    )
    draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)


def draw_boxes(image, boxes, labels, scores, keep, max_boxes=10, min_score=0.01, classes=None):
    """
    returns PIL image with boxes
    """
    colors = list(ImageColor.colormap.values())
    font = ImageFont.load_default()

    im_height, im_width = image.shape[1:]
    image_pil = T.ToPILImage()(image)

    for i in range(min(len(boxes), max_boxes)):
        if scores[i] >= min_score and i in keep:
            xmin, ymin, xmax, ymax = tuple(boxes[i])
            xmin, ymin, xmax, ymax = (
                xmin / im_width,
                ymin / im_height,
                xmax / im_width,
                ymax / im_height,
            )
            label = classes[labels[i].item()] if classes is not None else labels[i].item()
            display_str = f"{label}: {int(100 * scores[i])}%"
            color = colors[hash(labels[i]) % len(colors)]
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str=display_str)
    return image_pil


@torch.inference_mode()
def predict(img, model, device):
    model.eval()

    pred = model([img.to(device)])[0]

    keep = torchvision.ops.nms(pred["boxes"], pred["scores"], 0.1)
    return pred, keep


def predict_and_display(img, model, classes):
    """
    predicts boxes + labels and draws them on images
    """
    pred, keep = predict(img, model)
    return draw_boxes(
        img.cpu(), pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu(), keep.cpu(), classes=classes,
    )


def get_iou(box1, box2):
    """
    computes intersection over union
    boxes on form: [xmin, ymin, xmax, ymax]
    """
    bb1 = {
        "x1": box1[0],
        "y1": box1[1],
        "x2": box1[2],
        "y2": box1[3],
    }
    bb2 = {
        "x1": box2[0],
        "y1": box2[1],
        "x2": box2[2],
        "y2": box2[3],
    }

    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def compute_preds_for_classification(preds, targets):
    """
    Computes which predictions are for which targets.
    Returns only predictions which have a corresponding target
    Returns new list of predictions and targets

    preds: list of predictions
    targets: list of targets
    """
    new_preds = []
    new_targets = []

    targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
    preds = [{k: v.cpu() for k, v in t.items()} for t in preds]

    for pred, target in zip(preds, targets):
        new_pred = {"boxes": [], "labels": [], "index": []}
        new_target = {"boxes": [], "labels": [], "index": []}

        # loop over each prediction to find best match to target
        for j, box_p in enumerate(pred["boxes"]):

            candidates = []
            for i, box_t in enumerate(target["boxes"]):
                iou = get_iou(box_p, box_t)

                if iou >= 0.5:  # only keep detections of a target
                    candidates += [(i, iou)]

            # sort to find best matching target
            candidates = sorted(candidates, key=lambda x: -x[1])

            # did not detect a target if no candidates
            if len(candidates) > 0:
                i, _ = candidates[0]  # get best
                new_pred["boxes"] += [box_p]
                new_pred["labels"] += [pred["labels"][j]]
                new_pred["index"] += [j]

                new_target["boxes"] += [box_t]
                new_target["labels"] += [target["labels"][i]]
                new_target["index"] += [i]

        new_preds += [new_pred]
        new_targets += [new_target]

    return new_preds, new_targets


def crop_to_bbox(images, targets, img_size):
    cropped_images = []
    for img, detection in zip(images, targets):
        img = img.cpu()
        for box in detection["boxes"]:
            new_img = TF.resized_crop(
                img=img,
                top=int(box[0]),
                left=int(box[1]),
                height=int(box[2] - box[0]),
                width=int(box[3] - box[1]),
                size=(img_size, img_size),
            )
            cropped_images += [new_img]

    cropped_images = torch.stack(cropped_images)
    return cropped_images

