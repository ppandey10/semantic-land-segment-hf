from pathlib import Path
from landcover_segmentation.constants import *
from landcover_segmentation.utils.common import read_yaml, create_directories
from landcover_segmentation.entity import (
    DataIngestionConfig, 
    DataPreprocessingConfig, 
    DataLoaderConfig,
    ModelTrainerConfig
)

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


    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_path=config.data_path,
            PATCH_SIZE=config.PATCH_SIZE,
            patch_data_path=config.patch_data_path
        )

        return data_preprocessing_config
    

    def get_data_loader_config(self) -> DataLoaderConfig:
        config = self.config.data_loader
        params = self.params.SegmentationModelArguements

        # create directory
        create_directories([config.root_dir])

        data_loader_config = DataLoaderConfig(
            root_dir=config.root_dir,
            preprocessed_data_path=config.preprocessed_data_path,
            BACKBONE=params.BACKBONE,
            pretrained=params.pretrained,
            n_classes=params.n_classes,
            batch_size=params.batch_size
        )

        return data_loader_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.SMPTrainingArguments

        # Create directory
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            preprocessed_data_path=config.preprocessed_data_path,
            BACKBONE=params.BACKBONE,
            encoder_weights=params.encoder_weights,
            DEVICE=params.DEVICE,
            n_classes=params.n_classes,
            epochs=params.epochs,
            activation=params.activation,
            loss=params.loss,
            optimizer=params.optimizer,
            metrics=params.metrics
        )

        return model_trainer_config
