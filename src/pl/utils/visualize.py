import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.load_files import load_annotation, load_image

### TODO: show image id
def show_image(image):
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        assert Exception("Must be of type float32 or uint8 to show image")
    plt.figure(figsize=(15, 10), dpi=100)
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def visualize_with_anno(image_key, dataset_name):
    anno = load_annotation(image_key, dataset_name)
    image = load_image(image_key, dataset_name=dataset_name)

    for obj in anno["objects"]:
        xmin = int(obj["bbox"]["xmin"])
        ymin = int(obj["bbox"]["ymin"])
        xmax = int(obj["bbox"]["xmax"])
        ymax = int(obj["bbox"]["ymax"])
        label = str(obj["label"])

        image = draw_box(image, xmin, ymin, xmax, ymax, label)

    return image


def draw_box(image, xmin: int, ymin: int, xmax: int, ymax: int, text: str, color=(0, 0, 1)):

    image = np.ascontiguousarray(image, dtype=np.float32)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    font_thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    cv2.rectangle(image, (xmin, ymin), (xmin + text_w, ymin - text_h), color, -1)
    cv2.putText(
        image,
        text,
        (xmin, ymin),
        font,
        font_scale,
        (1, 1, 1),
        font_thickness,
        cv2.LINE_AA,
    )
    return image


def draw_boxes(image, boxes, labels, scores=None, min_score=0.0):
    for i in range(len(boxes)):
        if scores is not None:
            if scores[i] >= min_score:
                xmin, ymin, xmax, ymax = tuple(boxes[i])
                text = f"{labels[i]}: {int(scores[i] * 100)}"
                # color = colors[hash(class_names[i]) % len(colors)]
                image = draw_box(image, int(xmin), int(ymin), int(xmax), int(ymax), text)
        else:
            xmin, ymin, xmax, ymax = tuple(boxes[i])
            text = f"{labels[i]}"
            # color = colors[hash(class_names[i]) % len(colors)]
            image = draw_box(image, int(xmin), int(ymin), int(xmax), int(ymax), text)
    return image


if __name__ == "__main__":
    image_key = "_JJT0_vzPb4ooDz2GWUVrA"  # "_2xoR0ZHe-cO4fRlANU3wg"
    dataset_dir_name = "MTSD"

    # visualize traffic sign boxes on the image
    image = visualize_with_anno(image_key, dataset_dir_name)
    show_image(image)
