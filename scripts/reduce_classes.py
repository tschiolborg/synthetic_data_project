#
# this script is to filter classes in MTSD + creates a train-val-test split
#

import json
import os

import dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dotenv.load_dotenv(override=True)

# paths
DATA = os.getenv("DATA")
MTSD = os.getenv("MTSD")
if not DATA:
    raise Exception('Not able to find "DATA" environment variable')
if not MTSD:
    raise Exception('Not able to find "MTSD" environment variable')

classes_dir = os.path.join(DATA, "classes")
anno_dir = os.path.join(MTSD, "annotations")
print(f"classes dir: {classes_dir}")
print(f"annotation dir: {anno_dir}")


# load annotations
anno_files = os.listdir(anno_dir)
f"number of annotation files: {len(anno_files)}"


# load classes
with open(os.path.join(classes_dir, "classes.json")) as f:
    classes = json.load(f)
print(f"number of total classes: {len(classes)}")

# load classes to delete
with open(os.path.join(classes_dir, "delete.txt")) as f:
    delete = f.readlines()
delete = [s.replace("\n", "") for s in delete]
delete = [s for s in delete if s != ""]
print(f"number of classes to delete: {len(delete)}")


# create dir for new annotations
new_anno_dir = os.path.join(os.path.dirname(anno_dir), "anno_2")
os.makedirs(new_anno_dir, exist_ok=True)
print(f"New annotation dir: {new_anno_dir}")


# filter annotations with unwanted classes
n_pano = 0
n_in_delete = 0

for anno_file in anno_files:
    with open(os.path.join(anno_dir, anno_file)) as f:
        anno = json.load(f)

    if anno["ispano"]:
        n_pano += 1
        continue

    anno_new = {
        "labels": [],
        "boxes": [],
        "areas": [],
    }
    keep_anno = True

    for obj in anno["objects"]:
        if obj["label"] in delete:
            n_in_delete += 1
            keep_anno = False
            break

        anno_new["labels"].append(obj["label"])
        box = obj["bbox"]
        anno_new["boxes"].append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
        anno_new["areas"].append((box["xmax"] - box["xmin"]) * (box["ymax"] - box["ymin"]))

    if keep_anno:
        filename = os.path.join(new_anno_dir, anno_file)
        with open(filename, "w+") as f:
            json.dump(anno_new, f, indent=6)

print(f"number of panorama images removed: {n_pano}")
print(f"number of images removed with unwanted classes: {n_in_delete}")

anno_files = os.listdir(new_anno_dir)
print(f"number of images left: {len(anno_files)}")


# check images without any signs
n_no_signs = []

for anno_file in anno_files:
    try:
        with open(os.path.join(new_anno_dir, anno_file)) as f:
            anno = json.load(f)
        if len(anno["labels"]) == 0:
            n_no_signs += [anno_file]
    except Exception as e:
        print(f"Error file '{anno_file}' is incorrect format")

print(f"number of images without signs {len(n_no_signs)}")


# classes to keep annotations of
with open(os.path.join(classes_dir, "keep.txt")) as f:
    keep = f.readlines()
keep = [s.replace("\n", "") for s in keep]
keep = [s for s in keep if s != ""]
print(f"classes to keep: {len(keep)}")


# load classes mapping
with open(os.path.join(classes_dir, "map_mtsd2str.json")) as f:
    map_mtsd2str = json.load(f)
keep_str = list(set([map_mtsd2str[label] for label in keep]))


# count number of examples for each class and areas of signs
count = {}
areas = []

for anno_file in anno_files:
    with open(os.path.join(new_anno_dir, anno_file)) as f:
        anno = json.load(f)

    for label, area in zip(anno["labels"], anno["areas"]):
        if label in keep:
            areas.append(area)
            label = map_mtsd2str[label]
            if label in count:
                count[label] += 1
            else:
                count[label] = 1


# plot distribution of class sizes


def plot_count(x, y):
    """function to plot size of classes"""
    fig = plt.figure(figsize=(15, 10))
    sns.barplot(x=x, y=y, color="gray")
    plt.axvline(100, 0, 1, color="blue", label="100")
    plt.axvline(20, 0, 1, color="orange", label="20")
    plt.xlabel("Number of signs")
    plt.tight_layout()
    plt.legend()
    plt.show()


y, x = zip(*sorted(count.items(), key=lambda x: -x[1]))

plot_count(x=list(x), y=list(y))


print("classes with fewest exampls:")
print(sorted(count.items(), key=lambda x: x[1])[:5])


# plot distribution of sign areas

bins = list(range(10, 161, 10))

bin_labels = [str(i - 10) + "-" + str(i) for i in bins] + ["160<"]
bin_labels[0] = "<10"

bins = {((bin - 10) ** 2, bin ** 2): 0 for bin in bins}
bins[(160 ** 2, 1e8)] = 0

for c in areas:
    for bin in bins:
        if c >= bin[0] and c < bin[1]:
            bins[bin] += 1

x = list(bins.values())
y = [str(key) for key in bins.keys()]
cmap = ["red"] * 3 + ["gray"] * (len(x) - 3)

fig = plt.figure(figsize=(10, 7))
sns.barplot(y=y, x=x, palette=cmap)
plt.xlabel("Number of images")
plt.ylabel("Size of sign in pixels")
plt.yticks(np.arange(len(bin_labels)), bin_labels)
plt.tight_layout()
plt.show()

print("number of signs with area < 30^2 px")
print(sum(list(bins.values())[:3]))

print("number of signs with area >= 30^2 px")
print(sum(list(bins.values())[3:]))


# count number of examples for each class with area >= 30^2 px

count2 = {}
n_too_small = 0
threshold = 30 ** 2

for anno_file in anno_files:
    with open(os.path.join(new_anno_dir, anno_file)) as f:
        anno = json.load(f)

    for label, area in zip(anno["labels"], anno["areas"]):
        if label in keep:
            label = map_mtsd2str[label]
            if area < threshold:
                n_too_small += 1
            else:
                if label in count2:
                    count2[label] += 1
                else:
                    count2[label] = 1


print(f"number of examples too small: {n_too_small}")


# plot new distribution of class sizes

y, x = zip(*sorted(count2.items(), key=lambda x: -x[1]))
plot_count(x=list(x), y=list(y))

print("classes with fewest exampls:")
print(sorted(count2.items(), key=lambda x: x[1])[:5])


# make dir for new annotations
new_anno_dir2 = os.path.join(os.path.dirname(anno_dir), "anno_53")
os.makedirs(new_anno_dir2, exist_ok=True)
print(new_anno_dir2)

# load mapping from class str label to int label
with open(os.path.join(classes_dir, "map_str2num.json")) as f:
    map_str2num = json.load(f)


# filter images too small

n_img_too_small = 0
n_imgs = 0
threshold = 30 ** 2
num_labels = 53

for anno_file in anno_files:
    with open(os.path.join(new_anno_dir, anno_file)) as f:
        anno = json.load(f)

    anno_new = {
        "labels": [],
        "labels_str": [],
        "boxes": [],
        "areas": [],
    }
    keep_anno = False

    for label, box, area in zip(anno["labels"], anno["boxes"], anno["areas"]):
        if label in keep:
            if area >= threshold:
                keep_anno = True
            anno_new["labels"].append(map_str2num[map_mtsd2str[label]])
            anno_new["labels_str"].append(map_mtsd2str[label])
        else:
            anno_new["labels"].append(num_labels + 1)
            anno_new["labels_str"].append("other-sign")

        anno_new["boxes"].append(box)
        anno_new["areas"].append(area)

    if keep_anno:
        n_imgs += 1
        filename = os.path.join(new_anno_dir2, anno_file)
        with open(filename, "w+") as f:
            json.dump(anno_new, f, indent=6)
    else:
        n_img_too_small

print(f"number of final images: {n_imgs}")
print(f"number of images too small: {n_img_too_small}")


# perform train-val-test split


from sklearn.model_selection import train_test_split

anno_files = os.listdir(new_anno_dir2)
len(anno_files)


# load data
threshold = 30 ** 2
data = []

for anno_file in anno_files:
    with open(os.path.join(new_anno_dir2, anno_file)) as f:
        anno = json.load(f)

    for label, label_str, area in zip(anno["labels"], anno["labels_str"], anno["areas"]):
        if label_str != "other-sign" and area >= threshold:
            data.append([anno_file, int(label)])

print(f"number of annotations loaded for train-val-test split: {len(data)}")


# get ids as X and labels as Y
ids = []
labels = []
sorted_labels = [x[0] for x in sorted(count2.items(), key=lambda x: x[1])]
for id, label in data:
    if id not in ids:
        for label2 in sorted_labels:
            if label == map_str2num[label2]:
                ids.append(id)
                labels.append(label)
                break

# create dirs for each split
train_dir = os.path.join(os.path.dirname(anno_dir), "anno_train")
os.makedirs(train_dir, exist_ok=True)
val_dir = os.path.join(os.path.dirname(anno_dir), "anno_val")
os.makedirs(val_dir, exist_ok=True)
test_dir = os.path.join(os.path.dirname(anno_dir), "anno_test")
os.makedirs(test_dir, exist_ok=True)


# split data by stratifing labels. 60-20-20
X_train, X_test, y_train, y_test = train_test_split(ids, labels, stratify=labels, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5)

print("sizes of each split:")
print(f"train = {len(X_train)}, val = {len(X_val)}, test = {len(X_test)}")

print("percent size of each split:")
print(
    f"train = {len(X_train)/len(anno_files)},"
    + f"val = {len(X_val)/len(anno_files)}, test = {len(X_test)/len(anno_files)}"
)


# save data
train_files = set(X_train)
val_files = set(X_val)
test_files = set(X_test)

for anno_file in anno_files:
    with open(os.path.join(new_anno_dir2, anno_file)) as f:
        anno = json.load(f)

    if anno_file in train_files:
        with open(os.path.join(train_dir, anno_file), "w+") as f:
            json.dump(anno, f, indent=6)

    if anno_file in val_files:
        with open(os.path.join(val_dir, anno_file), "w+") as f:
            json.dump(anno, f, indent=6)

    if anno_file in test_files:
        with open(os.path.join(test_dir, anno_file), "w+") as f:
            json.dump(anno, f, indent=6)


# plot distribution of class sizes for each split


def count_labels(dir_path, files):
    count = {}

    for anno_file in files:
        with open(os.path.join(dir_path, anno_file)) as f:
            anno = json.load(f)

        for label, label_str, area in zip(anno["labels"], anno["labels_str"], anno["areas"]):
            if label_str != "other-sign" and area >= threshold:
                if label in count:
                    count[label] += 1
                else:
                    count[label] = 1

    return sorted(count.items(), key=lambda x: x[0])


count_train = count_labels(train_dir, train_files)
count_val = count_labels(val_dir, val_files)
count_test = count_labels(test_dir, test_files)


def plot_count2(count, title, color):
    x, y = zip(*count)
    fig = plt.figure(figsize=(10, 4))
    sns.barplot(x=list(x), y=list(y), color=color)
    plt.ylabel("Number of signs")
    plt.xlabel("Class index")
    plt.title(title + " class distribution")
    plt.tight_layout()
    plt.show()


counts = (count_train, count_val, count_test)
titles = ["Train", "Validation", "Test"]
colors = ["gray", "lightblue", "lightcoral"]

for i in range(3):
    print(sorted(counts[i], key=lambda x: x[1])[:5])
    plot_count2(counts[i], titles[i], colors[i])
