import os
from pathlib import Path

import albumentations as A
import cv2
import dotenv
import numpy as np
import torchvision.transforms as T

from src import augmentations

dotenv.load_dotenv(override=True)

TEMP = os.getenv("TEMP")
if not TEMP:
    raise Exception("Not able to find environment variable")

DATA = os.getenv("DATA")
if not DATA:
    raise Exception("Not able to find environment variable")


def get_mask(img):
    """
    Returns mask of img. Return has shape (H, W, 3)
    img: image (H, W, 4)
    """

    mask = np.zeros(img[:, :, :3].shape, np.uint8)
    mask[img[:, :, 3] > 0] = 255
    return mask


def intersects(box, boxes):
    """
    Returns if True if box itersects any box in boxes
    box: [x0, y0, x1, y1]
    boxes: list of bboxes [x0, y0, x1, y1]
    """

    for b in boxes:
        if (box[0] <= b[2] and box[2] >= b[0]) or (box[1] <= b[3] and box[3] >= b[1]):
            return True

    return False


def try_add_tmp(img, tmp, alpha=None, boxes=list()):
    """
    Adds template to image if it does not overlap with other boxes
    img: background image (H, W, 3)
    tmp: template (H, W, 4)
    alpha: value used to adjust img brightness, which now also is applied on tmp
    boxes: list of bboxes for other objects on image we do not want to paste on: [x0, y0, x1, y1]
    """

    mask = get_mask(tmp)
    tmp = cv2.cvtColor(tmp[:, :, :3], cv2.COLOR_BGR2RGB)

    # resize
    max_size = np.random.randint(30, 100)
    tmp = A.LongestMaxSize(max_size=max_size)(image=tmp)["image"]
    mask = A.LongestMaxSize(max_size=max_size)(image=mask)["image"]

    # brightness
    tmp, _, _ = augmentations.brightness(tmp, alpha=alpha, beta=0)

    # position
    max0 = img.shape[0] - tmp.shape[0]
    max1 = img.shape[1] - tmp.shape[1]

    pos0 = np.random.randint(0, max0 + 1)
    pos1 = np.random.randint(0, max1 + 1)

    # geometric transformations
    tmp, mask, p = augmentations.perspective(tmp, mask, pos1)
    tmp, mask, theta = augmentations.rotate(tmp, mask)
    tmp, mask, factor = augmentations.scale(tmp, mask, relative_x=2 * pos1 / img.shape[1])
    tmp, mask = augmentations.remove_padding(tmp, mask)  # remove padding

    # position
    y0 = pos0
    y1 = y0 + tmp.shape[0]
    x0 = pos1
    x1 = x0 + tmp.shape[1]

    box = [x0, y0, x1, y1]

    # check is in image
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None

    # check for intersection
    if intersects(box, boxes):
        return None

    # adjust brightness to background
    tmp = augmentations.brightness_adjust(tmp, mask, img[y0:y1, x0:x1])

    # noise
    tmp, noise = augmentations.uniform_noise(tmp)

    # paste
    img = augmentations.blend(tmp, mask, img, y0, x0)

    output = {"img": img, "box": box}

    return output


def generate_image(backgound_file, tmp_files):
    """
    Pastes templates onto image
    backgound_file: path to background image
    tmp_files: path to directory with template images
    """

    img = cv2.imread(backgound_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    crop_size = min(img.shape[0:2])
    img = A.CenterCrop(width=crop_size, height=crop_size)(image=img)["image"]
    img = A.Resize(width=1000, height=1000)(image=img)["image"]

    img, alpha, beta = augmentations.brightness(img)

    target = {"boxes": [], "labels": [], "areas": []}

    for i in range(5):
        tmp_file = os.path.join(TEMP, tmp_files[i])

        # load
        tmp = cv2.imread(tmp_file, cv2.IMREAD_UNCHANGED)

        label = int(Path(tmp_file).stem)

        output = None
        while output is None:
            output = try_add_tmp(img, tmp, alpha, target["boxes"])

        img = output["img"]
        box = output["box"]

        target["boxes"] += [box]
        target["labels"] += [label]
        target["areas"] += [(box[2] - box[0]) * (box[3] - box[1])]

    print(target)
    T.ToPILImage()(img).show()


tmp_files = sorted(os.listdir(TEMP))

backgound_file = os.path.join(DATA, "coco_test.jpg")

generate_image(backgound_file, tmp_files)
