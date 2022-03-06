import json
import os

import dotenv
import cv2
import numpy as np
import matplotlib.pyplot as plt

dotenv.load_dotenv(override=True)

ROOT = os.getenv("ROOT")
if not ROOT:
    raise 'Not able to find "ROOT" environment variable'


def load_annotation(image_key, dataset_dir):
    with open(os.path.join(ROOT, 'data', dataset_dir, 'annotations', f"{image_key}.json"), "r") as fid:
        anno = json.load(fid)
    return anno

def load_image(id, image_dir, file_extention='jpg'):
    image_path = os.path.join(image_dir, f'{id}.{file_extention}')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255
    return image

def visualize_with_anno(image, anno):
    for obj in anno["objects"]:
        xmin = int(obj["bbox"]["xmin"])
        ymin = int(obj["bbox"]["ymin"])
        xmax = int(obj["bbox"]["xmax"])
        ymax = int(obj["bbox"]["ymax"])
        class_name = obj["label"]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 2)

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, class_name, (xmin+5, ymin-5), font, 1, (0,0,0), 1, cv2.LINE_AA)

    return image


if __name__ == "__main__":
    image_key = "_2xoR0ZHe-cO4fRlANU3wg"
    dataset_dir = "MTSD"
    image_folder = "train"

    # load the annotation json
    anno = load_annotation(image_key, dataset_dir)

    # load image
    image_dir = os.path.join(ROOT, 'data', dataset_dir, 'images', f'{image_folder}')
    image = load_image(image_key, image_dir)

    # visualize traffic sign boxes on the image
    image = visualize_with_anno(image, anno)

    plt.figure(figsize=(10, 10), dpi=100)
    imgplot = plt.imshow(image)
    plt.show()