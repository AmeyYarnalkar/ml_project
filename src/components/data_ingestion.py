"""
Tasks:

1. Read the data from the source
2. Convert them into the train and test split
3. Output them as train and test csv files
"""

import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exceptions import CustomException


# configuration class
@dataclass
class DataIngestionConfig:
    
    base_path = Path().cwd()
    
    raw_data_path = base_path / "artifacts" / "raw.csv"
    train_data_path = base_path / "artifacts" / "train.csv"
    test_data_path = base_path / "artifacts" / "test.csv"
    
    source_data_path = base_path / "notebook" / "data" / "stud.csv"


class DataIngestion:
    
    def __init__(self):
        self.config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        
        try:
            
            logging.info("Starting data ingestion")

            # read dataset
            df = pd.read_csv(self.config.source_data_path)
            logging.info("Dataset loaded successfully")

            # create artifacts folder
            self.config.raw_data_path.parent.mkdir(parents=True, exist_ok=True)

            # save raw data
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # train test split
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info("Train and test datasets saved")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()