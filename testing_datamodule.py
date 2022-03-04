import hydra
import cv2

from src.datamodules import MtsdDataModule
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
    print(targets)

    # cv2.imshow(f"{idx}", image[0].numpy().permute(2, 1, 0))
    # print(target)

    # if cv2.waitKey(0) == ord("q"):
    #     break
