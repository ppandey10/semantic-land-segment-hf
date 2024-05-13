import os
os.environ['http_proxy'] = 'http://proxy1.bgc-jena.mpg.de:3128' 
os.environ['https_proxy'] = 'http://proxy1.bgc-jena.mpg.de:3128'

import urllib.request as rqst
import zipfile

from landcover_segmentation.logging import logger
from landcover_segmentation.utils.common import get_size

from landcover_segmentation.entity import DataIngestionConfig

class DataIngestion:
    def __init__(
        self,
        config: DataIngestionConfig
    ):
        self.config = config
        self.filename = config.local_data_path

    def download_data_from_url(self):
        if not os.path.exists(self.config.local_data_path):
            filename, headers = rqst.urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_path
            )
            logger.info(f'{filename} downloaded! Information: \n{headers}')

        else:
            logger.info(f'{self.filename} already exists!')

    def extract_zip_file(self):
        if not os.path.exists(self.config.unzip_dir):
            unzip_data_path = self.config.unzip_dir
            os.makedirs(unzip_data_path, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_data_path)   

            logger.info(f'{unzip_data_path} created!')

            # Delete the zip file after extraction
            os.remove(self.config.local_data_path)  
            logger.info(f'{self.config.local_data_path} deleted!')

        else:
            logger.info(f'{self.config.unzip_dir} already exits!')
            # # Delete the zip file after extraction
            # os.remove(self.config.local_data_path)
            # logger.info(f'{self.config.local_data_path} deleted!')