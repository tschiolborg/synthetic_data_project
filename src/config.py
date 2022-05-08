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


class ClassifierConfig:
    name: str
    img_size: int
    criterion: str
    optimizer: OptimizerConfing


@dataclass
class TrainingConfig:
    use_coco: bool
    only_detect: bool
    num_classes: int
    epochs: int
    criterion: str


@dataclass
class TransformsConfig:
    min_area: int
    img_size: int


@dataclass
class SubDatasetConfig:
    do_transforms: bool
    batch_size: int
    transforms: TransformsConfig


@dataclass
class DatasetConfig:
    name: str
    num_workers: int
    threshold: int
    keep_other: bool
    mtsd_labels: str
    train: SubDatasetConfig
    val: SubDatasetConfig
    test: SubDatasetConfig


@dataclass
class UtilsConfig:
    log_dir: str
    log_dir_json: str
    model_dir: str


@dataclass
class CheckpointConfig:
    resume: bool
    path: str


@dataclass
class Config:
    training: TrainingConfig
    optimizer: OptimizerConfing
    lr_scheduler: LrSchedulerConfig
    classifier: ClassifierConfig
    dataset: DatasetConfig
    utils: UtilsConfig
    checkpoint: CheckpointConfig


@dataclass
class ConfigTest:
    testing: TrainingConfig
    dataset: DatasetConfig
    model_dir: str
