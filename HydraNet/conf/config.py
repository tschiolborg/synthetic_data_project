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
class DataloaderConfig:
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
class TrainConfig:
    epochs: int
    lr: float
    dataloader: DataloaderConfig
    dataset: DatasetConfig


@dataclass
class DevConfig:
    dataset: DatasetConfig
    dataloader: DataloaderConfig


@dataclass
class PipelineConfig:
    train: TrainConfig
    dev: DevConfig
    model: ModelConfig
