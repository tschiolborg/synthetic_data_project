import json
import os
import sys

from src.utils.load_files import ROOT
from src.utils.visualize import show_image, visualize_with_anno


### put in vis


def show_class(file_in, class_key):
    with open(os.path.join(ROOT, "data", "classes", file_in)) as f:
        classes = json.load(f)

    if class_key in classes:
        images = list(classes[class_key]["images"])

        for image_key in images:
            image = visualize_with_anno(image_key, "MTSD")
            show_image(image)

            my_input = input("type [exit] for stop, otherwise next image: ")
            if my_input == "exit":
                break


if __name__ == "__main__":

    file_in = "classes.json"
    class_key = "information--motorway--g1" if len(sys.argv) < 2 else sys.argv[1]

    show_class(file_in, class_key)



