from landcover_segmentation.config.configuration import ConfigurationManager
from landcover_segmentation.components.data_ingestion import DataIngestion
from landcover_segmentation.logging import logger

class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data_from_url()
        data_ingestion.extract_zip_file()
