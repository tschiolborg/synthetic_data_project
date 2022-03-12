import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from visualize import show_image, visualize_with_anno

from load_files import ROOT, load_annotation


def generate_all_classes(filename, dataset_dir_name):
    """generates all classes mentioned in annotations and saves them as a json"""

    ## assert
    data_dir = os.path.join(ROOT, "data")
    dataset_dir = os.path.join(data_dir, dataset_dir_name)

    if not os.path.isdir(data_dir):
        raise Exception("Could not find specified data directory at: " + data_dir)

    ## create data
    classes = {"_": {"id": 0, "images": {}, "count": -1}}
    i = 1
    for item in tqdm(list(os.listdir(os.path.join(dataset_dir, "annotations")))):
        image_key = Path(item).stem
        anno = load_annotation(image_key, dataset_dir_name)

        for obj in anno["objects"]:
            label = obj["label"]
            if label not in classes:
                classes[label] = dict()
                classes[label]["id"] = i
                classes[label]["count"] = 1
                classes[label]["images"] = {image_key: 1}
                i += 1
            else:
                if image_key not in classes[label]["images"]:
                    classes[label]["images"][image_key] = 0
                classes[label]["count"] += 1
                classes[label]["images"][image_key] += 1

    save_classes(filename, data_dir, classes)


def generate_manually_chosen_classes(file_in, file_out, dataset_dir_name):
    """generate subset of classes based on which you choose"""

    data_dir = os.path.join(ROOT, "data")

    if not os.path.isdir(data_dir):
        raise Exception("Could not find specified data directory at: " + data_dir)

    with open(os.path.join(data_dir, "classes", file_in)) as f:
        classes = json.load(f)

    new_classes = dict()
    i = 1
    for label, data in classes.items():
        print(i, data["id"], label, data["count"])

        if label == "_":
            print("chosen")
            new_classes[label] = classes[label]
            continue

        images = list(data["images"])

        for image_key in images:
            if image == images[-1]:
                print("last image for this class!")

            image = visualize_with_anno(image_key, dataset_dir_name)
            show_image(image)

            my_input = input("Choose? [y] for yes, [n] for no, otherwise next image: ")
            if my_input == "y":
                print("chosen")
                new_classes[label] = classes[label]
                i += 1
                break
            if my_input == "n":
                print("not chosen")
                break

    save_classes(file_out, data_dir, new_classes)


def save_classes(filename, data_dir, classes):
    filename, file_extension = os.path.splitext(filename)
    print(file_extension)
    if file_extension in [".json", ""]:
        filename += ".json"
    else:
        raise Exception("File extension must be json")

    print("\nnumber of classes: ", len(classes))

    ## save file
    filename = os.path.join(data_dir, "classes", filename)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w+") as f:
        print(f"saving as: {filename}")
        json.dump(classes, f, indent=6)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Missing argument for dataset directory-name")
    if len(sys.argv) < 3:
        raise Exception("Missing argument for output filename")
    if len(sys.argv) < 4:
        print("Missing argument for input filename so creating all classes")
        generate_all_classes(sys.argv[2], sys.argv[1])
    else:
        generate_manually_chosen_classes(sys.argv[3], sys.argv[2], sys.argv[1])
