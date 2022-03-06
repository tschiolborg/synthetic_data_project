import json
import os
import json
import sys

from tqdm import tqdm
import dotenv

dotenv.load_dotenv(override=True)

ROOT = os.getenv("ROOT")
if not ROOT:
    raise 'Not able to find "ROOT" environment variable'


def generate_all_classes(file, dataset_dir_name):
    """generates all classes mentioned in annotations and saves them as a json"""

    ## assert
    dataset_dir = os.path.join(ROOT, "data", dataset_dir_name)
    anno_dir = os.path.join(dataset_dir, "annotations")

    if not os.path.isdir(dataset_dir):
        raise "Could not find specified data directory at: " + dataset_dir

    if not os.path.isdir(anno_dir):
        raise "Could not find annotation directory: " + anno_dir

    filename, file_extension = os.path.splitext(file)
    print(file_extension)
    if file_extension in [".json", ""]:
        filename += ".json"
    else:
        raise "File extension must be json"

    ## create data
    classes = {"_": {"id": 0, "images": {}, "count": -1}}
    i = 1
    for fname in tqdm(
        list(os.listdir(os.path.join(ROOT, "data", dataset_dir_name, "annotations")))
    ):
        with open(os.path.join(anno_dir, fname), "r") as fid:
            anno = json.load(fid)
            for obj in anno["objects"]:
                label = obj["label"]
                if label not in classes:
                    classes[label] = dict()
                    classes[label]["id"] = i
                    classes[label]["count"] = 1
                    classes[label]["images"] = {fname: 1}
                    i += 1
                else:
                    if fname not in classes[label]["images"]:
                        classes[label]["images"][fname] = 0
                    classes[label]["count"] += 1
                    classes[label]["images"][fname] += 1

    print("\nnumber of classes: ", len(classes))

    ## save file
    filename = os.path.join(dataset_dir, "classes", filename)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w+") as f:
        print(f"saving as: {filename}")
        json.dump(classes, f, indent=6)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise "Missing argument for filename"
    if len(sys.argv) < 3:
        print("Missing argument for dataset directory-name so using 'data/annotations'")
        dir_name = ""
    else:
        dir_name = sys.argv[2]

    generate_all_classes(sys.argv[1], dir_name)
