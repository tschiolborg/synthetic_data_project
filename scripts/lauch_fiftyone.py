import fiftyone as fo
import fiftyone.zoo as foz
from tqdm import tqdm

coco_classes_path = r"C:\Users\saibo\Documents\bachelor_project\data\classes\coco_classes.txt"
with open(coco_classes_path) as f:
    classes = [item.replace("\n", "") for item in f.readlines()]

unwanted_classes = [
    "traffic light",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "fire hydrant",
    "stop sign",
    "parking meter",
]

# download data
# dataset = foz.load_zoo_dataset(
#     "coco-2017",
#     split="train",
#     label_types=["detections"],
#     max_samples=20_000,
# )


dataset_dir = r"C:\Users\saibo\fiftyone\coco-2017\train"
dataset_type = fo.types.COCODetectionDataset

# mixed images
mixed_dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir, dataset_type=dataset_type, name="mixed"
)


# unwanted images
unwanted_dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir, dataset_type=dataset_type, name="unwanted", classes=unwanted_classes
)

unwanted_images = [sample.filepath for sample in unwanted_dataset]


# filter such that final dataset does not contain unwanted ids
final_dataset = fo.Dataset("final_dataset")
for sample in tqdm(mixed_dataset):
    if sample.filepath not in unwanted_images:
        final_dataset.add_sample(sample)

print(f"total:     {len(mixed_dataset)}")
print(f"unwanted:  {len(unwanted_dataset)}")
print(f"final:     {len(final_dataset)}")


export_dir = r"C:\Users\saibo\Documents\bachelor_project\data\COCO"

# Export the dataset
final_dataset.export(
    export_dir=export_dir, dataset_type=dataset_type, label_field="ground_truth",
)

if __name__ == "__main__":
    session = fo.launch_app(final_dataset)
    session.wait()
