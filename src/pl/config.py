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
    only_detect: bool
    num_classes: int
    debug: bool
    gpus: int
    max_epochs: int


@dataclass
class DatasetConfig:
    dir: str
    batch_size: int
    num_workers: int
    transforms: str


@dataclass
class DatamoduleConfig:
    train: DatasetConfig
    val: DatasetConfig
    test: DatasetConfig
    anno_file: str


@dataclass
class UtilsConfig:
    log_dir: str
    model_dir: str


@dataclass
class Config:
    training: TrainingConfig
    optimizer: OptimizerConfing
    lr_scheduler: LrSchedulerConfig
    dataset: DatamoduleConfig
    utils: UtilsConfig
