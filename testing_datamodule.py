import hydra
import cv2

from src.datamodules import MtsdDataModule
from src.utils.visualize import show_image, insert_box
from transforms import get_transform

with hydra.initialize(config_path="conf"):
    cfg = hydra.compose(config_name="config.yaml")

    m = MtsdDataModule(cfg=cfg)
    m.setup()


print("\ntrain")
data_loader = m.train_dataloader()


for idx, (images, targets) in enumerate(data_loader):
    print(idx)
    print(images.shape)

    for image, target in zip(images, targets):
        image = image.permute(2, 1, 0).numpy()

        for label, boxes in zip(target["labels"], target["boxes"]):
            image = insert_box(image, boxes, label)

        show_image(image)

    if idx == 2:
        break
