import json
import os

import dotenv
import cv2
import numpy as np

dotenv.load_dotenv(override=True)

ROOT = os.getenv("ROOT")
if not ROOT:
    raise Exception('Not able to find "ROOT" environment variable')


def load_annotation(image_key, dataset_name):
    with open(
        os.path.join(ROOT, "data", dataset_name, "annotations", f"{image_key}.json"), "r"
    ) as fid:
        anno = json.load(fid)
    return anno


def load_image(id, image_dir=None, dataset_name=None, file_extention="jpg"):
    if image_dir is None:
        if dataset_name is not None:
            image_dir = get_image_dir(id, dataset_name)
        else:
            raise Exception("Cannot find directory (missing dataset_name or image_dir)")

    image_path = os.path.join(image_dir, f"{id}.{file_extention}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255
    return image


def get_image_dir(id, dataset_name):
    dataset_dir = os.path.join(ROOT, "data", dataset_name)

    images_dir = None

    for dir in ("train", "val", "test"):
        with open(os.path.join(dataset_dir, "splits", f"{dir}.txt")) as f:
            if id in f.read().splitlines():
                images_dir = dir

    if images_dir is None:
        raise Exception("Could not find image in any splits")

    return os.path.join(dataset_dir, "images", images_dir)
