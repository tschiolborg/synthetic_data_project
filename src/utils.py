import json
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageColor, ImageFont
import torchvision.transforms as T
import torch

from .engine import predict
from .config import Config, ConfigTest
from .datasets import MTSD_Dataset, GTSDB_Dataset
from .transforms import Transforms

import dotenv

dotenv.load_dotenv(override=True)


def load_data(cfg: Config):
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
            raise Exception('Not able to find "MTSD" environment variable')

        dataset_train = GTSDB_Dataset(
            os.path.join(GTSDB, "train"),
            GTSDB,
            transforms=my_transforms.get_transform_gtsdb(True),
            only_detect=cfg.training.only_detect,
        )
        dataset_val = GTSDB_Dataset(
            os.path.join(GTSDB, "test"),
            GTSDB,
            transforms=my_transforms.get_transform_gtsdb(False),
            only_detect=cfg.training.only_detect,
        )
    else:
        raise Exception(f"error cannot find dataset: {cfg.dataset.name}")

    return dataset_train, dataset_val


def load_data_test(cfg: ConfigTest):
    my_transforms = Transforms(
        min_area_val=cfg.dataset.test.transforms.min_area,
        img_size_val=cfg.dataset.test.transforms.img_size,
    )

    if cfg.dataset.name == "MTSD":

        MTSD = os.getenv("MTSD")
        if not MTSD:
            raise Exception('Not able to find "MTSD" environment variable')

        img_dir = os.path.join(MTSD, "images")
        anno_test = os.path.join(MTSD, "anno_val")

        dataset_test = MTSD_Dataset(
            img_dir,
            anno_test,
            transforms=my_transforms.get_transform(False),
            only_detect=cfg.testing.only_detect,
        )

    elif cfg.dataset.name == "GTSDB":

        GTSDB = os.getenv("GTSDB")
        if not GTSDB:
            raise Exception('Not able to find "MTSD" environment variable')

        dataset_test = GTSDB_Dataset(
            os.path.join(GTSDB, "images"),
            GTSDB,
            transforms=my_transforms.get_transform_gtsdb(False),
            only_detect=cfg.testing.only_detect,
        )
    else:
        raise Exception(f"error cannot find dataset: {cfg.dataset.name}")

    return dataset_test


def load_optimizer(cfg: Config, params):
    if cfg.optimizer.name == "SGD":
        return torch.optim.SGD(
            params,
            lr=cfg.optimizer.params["lr"],
            momentum=cfg.optimizer.params["momentum"],
            weight_decay=cfg.optimizer.params["weight_decay"],
        )
    else:
        raise Exception(f"error cannot find optimizer: {cfg.optimizer.name}")


def load_lr_scheduler(cfg: Config, optimizer):
    if cfg.lr_scheduler.name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.lr_scheduler.params["step_size"],
            gamma=cfg.lr_scheduler.params["gamma"],
        )
    else:
        raise Exception(f"error cannot find optimizer: {cfg.optimizer.name}")


class Json_writer:
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


def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str=""
):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
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
        [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
        fill=color,
    )
    draw.text(
        (left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font
    )


def draw_boxes(image, boxes, labels, scores, keep, max_boxes=10, min_score=0.01, classes=None):
    """ returns PIL image with boxes """
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
            draw_bounding_box_on_image(
                image_pil, ymin, xmin, ymax, xmax, color, font, display_str=display_str
            )
    return image_pil


def predict_and_display(img, model, classes):
    pred, keep = predict(img, model)
    return draw_boxes(
        img.cpu(),
        pred["boxes"].cpu(),
        pred["labels"].cpu(),
        pred["scores"].cpu(),
        keep.cpu(),
        classes=classes,
    )
