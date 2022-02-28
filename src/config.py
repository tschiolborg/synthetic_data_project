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
    shuffle: bool
    batch_size: int
    num_workers: int


@dataclass
class BackboneConfig:
    name: str
    fpn_compatible: bool


@dataclass
class ModelConfig:
    name: str
    num_classes: int
    weights: str


@dataclass
class TrainerConfig:
    min_epochs: int
    max_epochs: int
    lr: float
    gpus: int


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

