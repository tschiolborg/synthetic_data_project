#
# reduces the annotations size to given percent while keeping small labels
#   args:
#       size_percent (float): percent of desired size of annotations
#

import os
import sys
import json
import random

import dotenv

dotenv.load_dotenv(override=True)

DATA = os.getenv("DATA")
MTSD = os.getenv("MTSD")
if not DATA:
    raise Exception('Not able to find "DATA" environment variable')
if not MTSD:
    raise Exception('Not able to find "MTSD" environment variable')


def load_annos() -> dict:
    """load annotations"""

    anno_files = os.listdir(os.path.join(MTSD, "anno_train"))

    annos = {}
    for filename in anno_files:
        with open(os.path.join(MTSD, "anno_train", filename)) as f:
            anno = json.load(f)
            idx_keep = {i for i, label in enumerate(anno["labels"]) if label != 54}
            anno["labels"] = [label for i, label in enumerate(anno["labels"]) if i in idx_keep]
            anno["labels_str"] = [
                label for i, label in enumerate(anno["labels_str"]) if i in idx_keep
            ]
            anno["boxes"] = [label for i, label in enumerate(anno["boxes"]) if i in idx_keep]
            anno["areas"] = [label for i, label in enumerate(anno["areas"]) if i in idx_keep]
            annos[filename] = anno

    print(f"number of annotations loaded: {len(annos)}\n")

    return annos


def count_labels(annos: dict) -> dict:
    """count number of each label"""

    count = {}

    for anno in annos.values():
        for label in anno["labels"]:
            if label in count:
                count[label] += 1
            else:
                count[label] = 1

    count = {key: count[key] for key in sorted(count, key=lambda x: count[x])}

    print("Label sizes before:")
    print(count, "\n")

    return count


def reduce_annos(size_percent: float, annos: dict, count: dict) -> dict:
    """reduce annotation dict to smaller size while keeping small classes"""

    size = int(len(annos) * size_percent) + 1
    new_annos = {}
    new_count = {key: 0 for key in count.keys()}

    anno_copy = annos.copy()

    l = list(anno_copy.items())
    random.shuffle(l)
    anno_copy = dict(l)

    c = 1
    while len(new_annos) <= size:  # loop until right size

        # loop over annos
        for key, anno in list(anno_copy.items()):
            seen_labels = []
            chosen = False

            for label in anno["labels"]:
                seen_labels += [label]  # collect labels

                if new_count[label] < c:  # add if label is rare
                    chosen = True

            if chosen:
                new_annos[key] = anno
                for label in seen_labels:
                    new_count[label] += 1
                del anno_copy[key]  # remove
        c += 1

    print(f"size given: {size_percent*100:.2f}%")
    print(f"final size: {len(new_annos)}, which is {len(new_annos)/len(annos)*100:.2f}%\n")

    print("Label sizes after:")
    print(new_count, "\n")

    return new_annos


def save_annos(annos: dict, new_dir: str) -> None:
    """save to new dir"""

    path = os.path.join(MTSD, new_dir)
    os.makedirs(path, exist_ok=True)

    for name, anno in annos.items():
        with open(os.path.join(path, name), "w+") as f:
            json.dump(anno, f, indent=6)


def main(size_percent: float) -> None:

    annos = load_annos()
    count = count_labels(annos)

    annos_reduced = reduce_annos(size_percent, annos, count)

    new_dir = f"anno_{size_percent}"
    save_annos(annos_reduced, new_dir)

    print("done")


if __name__ == "__main__":

    size_percent = 0.02 if sys.argv == 1 else float(sys.argv[1])
    main(size_percent)

