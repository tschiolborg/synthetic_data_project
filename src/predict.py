import os
import warnings

from omegaconf import DictConfig
import hydra
import torch

from src.utils.load_files import ROOT
from src.models.faster_rcnn_module import FasterRCNNLitningModule
from src.datamodules.mtsd_datamodule import MtsdDataModule
from src.utils.visualize import draw_boxes, show_image

warnings.filterwarnings("ignore")


def predict(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model
    Args:
        cfg: hydra config
    """
    # set_seed(cfg.training.seed)

    # path = os.path.join(ROOT, "outputs", "2022-03-14", "20-59-42", "20-59-42.pth")
    path = os.path.join(
        ROOT,
        "outputs",
        "2022-03-22",
        "19-35-15",
        "default",
        "8",
        "checkpoints",
        "epoch=9-step=9679.ckpt",
    )

    datamodule = MtsdDataModule(cfg=cfg)
    datamodule.setup()
    test_loader = datamodule.val_dataloader()

    # model = MtsdLitModule(cfg=cfg)
    # model.load_state_dict(torch.load(path))
    model = FasterRCNNLitningModule.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
    model.eval()

    with torch.no_grad():
        for images, targets in test_loader:
            print(targets)

            _, preds = model(images, targets)

            print(preds)

            for img, pred in zip(images, preds):
                img = img.cpu().permute((1, 2, 0))
                img = draw_boxes(img, pred["boxes"], pred["labels"], pred["scores"])
                show_image(img)


@hydra.main(config_path=os.path.join(ROOT, "conf/"), config_name="config.yaml")
def run_model(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    run_model()
