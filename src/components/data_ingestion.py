import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Configuration
    """
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv') 
    train_data_path: str = os.path.join('artifacts', 'train.csv') 
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    split_ratio: float = 0.2

class DataIngestion:
    """
    Data Ingestion Class
    """
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the data from the given path
            df = pd.read_csv('notebook/data/Stud.csv')

            # Check if the DataFrame is empty
            if df.empty:
                raise ValueError("The DataFrame is empty. Please check the input data.")
            logging.info("Data read successfully as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test sets to the specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion completed successfully")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path 
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, extra = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))