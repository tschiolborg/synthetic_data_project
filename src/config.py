from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DatasetConfig:
    use_single_annotation_file: bool
    anno_dir: str
    img_dir: str
    classes: List[str]
    label_map: Dict[str, int]


@dataclass
class DatamoduleConfig:
    dataset: DatasetConfig
    shuffle: bool
    batch_size: int
    num_workers: int


@dataclass
class ModelConfig:
    num_classes: int
    lr: float


@dataclass
class TrainerConfig:
    gpus: int
    max_epochs: int


@dataclass
class Config:
    model = ModelConfig
    datamodule: DatamoduleConfig
    trainer: TrainerConfig




'''

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# Model definitition
cs.store(group="model", name="simple_mlp", node=ObjectConf)

# Dataset definitition
cs.store(group="dataset", name="mnist", node=DatasetConf)

# Optimizers definitition
cs.store(group="optimizers", name="adam", node=OptimizerConf)

# Training definitition
cs.store(group="trainer", name="debugging", node=ObjectConf)

# Logging Entities
cs.store(group="loggers", name="thomas-chaton", node=LoggersConf)


'''

