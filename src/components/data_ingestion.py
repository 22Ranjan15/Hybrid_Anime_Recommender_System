import os
import sys
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.exception import CustomException
from config.paths_config import *
from utils.utils import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config['bucket_name']
        self.file_names = self.config['bucket_file_names']

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion process initialized with bucket: {self.bucket_name} and files: {self.file_names}")

    def download_from_gcp(self):
        try:
            logger.info(f"Attempting to download {self.file_names} from bucket {self.bucket_name}")
            
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                blob = bucket.blob(file_name)
                blob.download_to_filename(file_path)    # Downloading the file

                if file_name == "animelist.csv":
                    try:
                        data = pd.read_csv(file_path, nrows=5000000)   # Reading first 5 million rows
                        
                        # Checking if the file already exists in the directory
                        if os.path.exists(file_path):
                            os.remove(file_path)    # If exists, then delete

                        data.to_csv(file_path, index=False) # Saving first 5 million rows 
                        logger.info(f"Processed and saved first 5 million rows of '{file_name}'.")

                    except Exception as e:
                        logger.error(f"Error processing '{file_name}': {e}")
                else:
                    logger.info(f"Downloaded '{file_name}' from bucket '{self.bucket_name}'.")

        except Exception as e:
            raise CustomException(f"Failed to download file '{self.file_names}' from bucket '{self.bucket_name}'. Error: {str(e)}", sys) from e


    def run(self):
        try:
            logger.info(f"Starting the Data Ingestion process for bucket '{self.bucket_name}' with the following files: {self.file_names}.")
            self.download_from_gcp()
            logger.info("Data Ingestion process completed successfully.")
        
        except Exception as e:
            raise CustomException(f"Data Ingestion failed due to an error: {str(e)}", sys) from e


if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
        data_ingestion.run()

    except Exception as e:
        logger.error(f"Data Ingestion failed: {e}")