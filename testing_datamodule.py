import hydra

from src.datamodules import MtsdDataModule

with hydra.initialize(config_path="conf"):
    cfg = hydra.compose(config_name="config.yaml")

    m = MtsdDataModule(cfg=cfg)
    m.setup()


print("\ntrain")
data_loader = m.train_dataloader()

for data in data_loader:
    print(data)
