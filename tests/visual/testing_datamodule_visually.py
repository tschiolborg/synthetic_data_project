import os

import numpy as np
import hydra

from src.datamodules import MtsdDataModule
from src.utils.visualize import show_image, insert_box
from src.utils.load_files import ROOT


if __name__ == "__main__":
    conf_dir = os.path.join(ROOT, "conf")

    with hydra.initialize_config_dir(config_dir=conf_dir):
        cfg = hydra.compose(config_name="debug.yaml")

    datamodule = MtsdDataModule(cfg=cfg)
    datamodule.setup()

    data_loaders = (datamodule.train_dataloader(), datamodule.val_dataloader())

    for name, data_loader in zip(("train", "val"), data_loaders):
        images, targets = next(iter(data_loader))

        print(name)
        print(images.shape)

        for image, target in zip(images, targets):
            image = image.permute(1, 2, 0).numpy()

            for label, boxes in zip(target["labels"].numpy(), target["boxes"]):
                image = insert_box(image, boxes, label)

            show_image(image)
