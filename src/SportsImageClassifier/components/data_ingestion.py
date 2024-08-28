import os
import kaggle
import zipfile
from pathlib import Path
from SportsImageClassifier import logger
from SportsImageClassifier.utils.common import get_size
from SportsImageClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.unzip_dir
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            kaggle.api.dataset_download_files('gpiosenka/sports-classification', Path(zip_download_dir), unzip=True)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

