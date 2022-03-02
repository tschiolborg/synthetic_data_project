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


def gen_classes(file):

    filename, file_extension = os.path.splitext(file)
    print(file_extension)
    if file_extension in [".json", ""]:
        filename += ".json"
    else:
        raise "File extension must be json"

    labels = set()
    for fname in tqdm(list(os.listdir(os.path.join(ROOT, "data", "annotations")))):
        with open(os.path.join("data", "annotations", fname), "r") as fid:
            anno = json.load(fid)
            for obj in anno["objects"]:
                label = obj["label"]
                labels.add(label)

    classes_dict = {val: i + 1 for i, val in enumerate(labels)}
    classes_dict["_"] = 0

    print("\nnumber of classes: ", len(classes_dict))

    filename = os.path.join(ROOT, "data", "classes", filename)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w+") as f:
        print(f"saving as: {filename}")
        json.dump(classes_dict, f, indent=6)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise "Missing argument for filename"

    gen_classes(sys.argv[1])
