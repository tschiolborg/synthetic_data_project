import json
import os

import dotenv
import cv2
import numpy as np

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


