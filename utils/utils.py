import os
import pandas as pd
from src.logger import get_logger
from src.exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            logger.error(f"YAML file not found at path: {file_path}")
            logger.debug(f"Config content: {config}")
            raise FileNotFoundError(f"YAML file not found at the specified path: {file_path}")
        
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"Successfully read the YAML file from path: {file_path}")
            return config
    except Exception as e:
        logger.error(f"Unexpected error while reading YAML file at path: {file_path}. Error: {e}")
        raise CustomException(f"An unexpected error occurred while reading the YAML file at path: {file_path}", e)
    

