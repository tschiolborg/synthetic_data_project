import logging

import hydra
import torch
import torchvision

from HydraNet.conf.config import MLConfig

log = logging.getLogger(__name__)


@hydra.main(config_path="HydraNet/conf", config_name="config")
def main(cfg: MLConfig):

    log.info("Running main.py")
    print(cfg.train.path)

    resnet = torchvision.models.resnet50(pretrained=True)


if __name__ == "__main__":
    main()
