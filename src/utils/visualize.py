import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from load_files import load_annotation, load_image, ROOT


def show_image(image):
    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow((image * 255).astype(np.uint8))
    plt.show()


def visualize_anno(image_key, dataset_dir_name):
    anno = load_annotation(image_key, dataset_dir_name)
    image = load_image(image_key, dataset_dir_name=dataset_dir_name)

    for obj in anno["objects"]:
        xmin = int(obj["bbox"]["xmin"])
        ymin = int(obj["bbox"]["ymin"])
        xmax = int(obj["bbox"]["xmax"])
        ymax = int(obj["bbox"]["ymax"])
        class_name = obj["label"]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, class_name, (xmin + 5, ymin - 5), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    return image


if __name__ == "__main__":
    image_key = "_2xoR0ZHe-cO4fRlANU3wg"
    dataset_dir_name = "MTSD"

    # visualize traffic sign boxes on the image
    image = visualize_anno(image_key, dataset_dir_name)
    show_image(image)
