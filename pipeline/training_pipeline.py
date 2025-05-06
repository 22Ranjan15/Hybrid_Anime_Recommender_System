from utils.utils import read_yaml
from config.paths_config import *
from src.components.data_processer import DataProcessor
from src.components.model_trainer import ModelTraining

if __name__=="__main__":
    data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
    data_processor.run()

    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()

