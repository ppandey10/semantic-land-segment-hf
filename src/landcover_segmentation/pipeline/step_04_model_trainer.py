from landcover_segmentation.config.configuration import ConfigurationManager
from landcover_segmentation.components.model_trainer import ModelTrainer
from landcover_segmentation.logging import logger

class ModelTrainerPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.model_trainer()
        model_trainer.save_model()