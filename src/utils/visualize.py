import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.load_files import load_annotation, load_image, ROOT


def show_image(image):
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.unit8:
        assert Exception("Must be of type float32 or uint8 to show image")
    plt.figure(figsize=(10, 10), dpi=100)
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def visualize_with_anno(image_key, dataset_name):
    anno = load_annotation(image_key, dataset_name)
    image = load_image(image_key, dataset_name=dataset_name)

    for obj in anno["objects"]:
        xmin = obj["bbox"]["xmin"]
        ymin = obj["bbox"]["ymin"]
        xmax = obj["bbox"]["xmax"]
        ymax = obj["bbox"]["ymax"]
        label = obj["label"]

        image = insert_box(image, (xmin, ymin, xmax, ymax), label)

    return image


def insert_box(image, box, label):
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])

    image = np.ascontiguousarray(image, dtype=np.float32)
    label = str(label)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(image, label, (xmin + 5, ymin - 5), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    return image


if __name__ == "__main__":
    image_key = "_2xoR0ZHe-cO4fRlANU3wg"
    dataset_dir_name = "MTSD"

    # visualize traffic sign boxes on the image
    image = visualize_with_anno(image_key, dataset_dir_name)
    show_image(image)
