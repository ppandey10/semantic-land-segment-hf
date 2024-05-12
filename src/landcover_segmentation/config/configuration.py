from pathlib import Path
from landcover_segmentation.constants import *
from landcover_segmentation.utils.common import read_yaml, create_directories
from landcover_segmentation.entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath: Path = CONFIG_FILEPATH,
        params_filepath: Path = PARAMS_FILEPATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_path=config.local_data_path,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
