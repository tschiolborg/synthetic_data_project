import os
import warnings

from omegaconf import DictConfig
import hydra
import torch

from src.utils.load_files import ROOT
from src.models.mtsd_module import MtsdLitModule
from src.datamodules.mtsd_datamodule import MtsdDataModule
from src.utils.visualize import insert_box, show_image

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
        "2022-03-15",
        "18-23-56",
        "default",
        "3",
        "checkpoints",
        "epoch=4-step=2609.ckpt",
    )

    datamodule = MtsdDataModule(cfg=cfg)
    datamodule.setup()
    test_loader = datamodule.val_dataloader()

    # model = MtsdLitModule(cfg=cfg)
    # model.load_state_dict(torch.load(path))
    model = MtsdLitModule.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
    model.eval()

    with torch.no_grad():
        for images, targets in test_loader:
            preds = model(images, targets)

            print(targets)
            print(preds)
            for img, target, pred in zip(images, targets, preds):
                img = img.cpu().permute((1, 2, 0))
                for box in target["boxes"]:
                    img = insert_box(img, box.cpu(), "1")
                for box in pred["boxes"]:
                    img = insert_box(img, box.cpu(), "2")
                show_image(img)


@hydra.main(config_path="conf/", config_name="config.yaml")
def run_model(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    run_model()