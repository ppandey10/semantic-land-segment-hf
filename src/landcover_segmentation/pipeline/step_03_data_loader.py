from landcover_segmentation.config.configuration import ConfigurationManager
from landcover_segmentation.components.data_loader import SegDataLoader
from landcover_segmentation.logging import logger

class DataLoaderPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_loader_config = config.get_data_loader_config()
        data_loader = SegDataLoader(config=data_loader_config)
        train_generator = data_loader.TrainGenerator(data_type='train')
        val_generator = data_loader.TrainGenerator(data_type='val')
        test_generator = data_loader.TrainGenerator(data_type='test')