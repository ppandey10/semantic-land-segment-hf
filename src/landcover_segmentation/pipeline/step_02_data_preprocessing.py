from landcover_segmentation.config.configuration import ConfigurationManager
from landcover_segmentation.components.data_preprocessing import DataPreprocessing
from landcover_segmentation.logging import logger

class DataPreprocessingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.create_patch_folder()
        data_preprocessing.generate_image_patches()
        data_preprocessing.generate_mask_patches()
        data_preprocessing.remove_unlabelled_patches()
        data_preprocessing.split_dataset()
        data_preprocessing.restructure_data_folder()