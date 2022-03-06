import os

import cv2
import matplotlib.pyplot as plt

from load_files import load_annotation, load_image, ROOT


def show_image(image):
    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(image)
    plt.show()


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
    show_image(image)

