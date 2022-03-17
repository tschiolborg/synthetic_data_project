import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from src.datamodules.mtsd_datamodule import MtsdDataModule
from src.models.mtsd_module import MtsdLitModule

warnings.filterwarnings("ignore")


def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model
    Args:
        cfg: hydra config
    """
    # set_seed(cfg.training.seed)
    # hparams = flatten_omegaconf(cfg)

    datamodule = MtsdDataModule(cfg=cfg)
    model = MtsdLitModule(cfg=cfg)

    # early_stopping = pl.callbacks.EarlyStopping(**cfg.callbacks.early_stopping.params)
    model_checkpoint = pl.callbacks.ModelCheckpoint(**cfg.model_checkpoint.params)
    # lr_logger = pl.callbacks.LearningRateLogger()

    tb_logger = TensorBoardLogger(save_dir=cfg.save_dir)

    trainer = pl.Trainer(
        logger=[tb_logger],
        # early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint,
        # callbacks=[lr_logger],
        **cfg.trainer,
    )
    trainer.fit(model=model, datamodule=datamodule)

    # save as a simple torch model
    model_name = os.getcwd().split("\\")[-1] + ".pth"
    print(model_name)
    torch.save(model.model.state_dict(), model_name)


@hydra.main(config_path="conf/", config_name="config.yaml")
def run_model(cfg: DictConfig) -> None:
    print(cfg)
    run(cfg)


if __name__ == "__main__":
    run_model()
