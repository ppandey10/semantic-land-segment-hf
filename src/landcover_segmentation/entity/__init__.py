from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_path: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    STATUS_FILE: str
    data_path: Path
    PATCH_SIZE: int
    patch_data_path: Path

@dataclass(frozen=True)
class DataLoaderConfig:
    root_dir: Path
    preprocessed_data_path: Path
    BACKBONE: str
    pretrained: str
    n_classes: int
    batch_size: int