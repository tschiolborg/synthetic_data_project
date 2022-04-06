from dataclasses import dataclass
from typing import Dict


@dataclass
class OptimizerConfing:
    name: str
    params: Dict


@dataclass
class LrSchedulerConfig:
    name: str
    params: Dict


@dataclass
class TrainingConfig:
    use_coco: bool
    only_detect: bool
    num_classes: int
    epochs: int


@dataclass
class TransformsConfig:
    min_area: int
    img_size: int


@dataclass
class SubDatasetConfig:
    batch_size: int
    transforms: TransformsConfig


@dataclass
class DatasetConfig:
    name: str
    num_workers: int
    train: SubDatasetConfig
    val: SubDatasetConfig
    test: SubDatasetConfig
    

@dataclass
class UtilsConfig:
    log_dir: str
    model_dir: str


@dataclass
class Config:
    training: TrainingConfig
    optimizer: OptimizerConfing
    lr_scheduler: LrSchedulerConfig
    dataset: DatasetConfig
    utils: UtilsConfig

