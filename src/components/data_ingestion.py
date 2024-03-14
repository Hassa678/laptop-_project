import sys
import os
# Get the parent directory of the current file (src)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from exception import CustomException
from logger import logging
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer,ModelTrainingConfig

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

@dataclass
class DataIngestionConfig:
    train_data_path = str=os.path.join('artifact','train_data.csv')
    test_data_path = str=os.path.join('artifact','test_data.csv')
    raw_data_path = str=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv('notebook\data\laptopData.csv')
            df = df.drop('Unnamed: 0',axis=1)
            df = df.drop_duplicates()
            df.reset_index(drop=True, inplace=True)
            df['Weight'] = df['Weight'].replace('?', np.nan)
            imputer.fit(df[['Price']])
            df['Price'] = imputer.transform(df[['Price']])
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initialized")
            train_set,test_set = train_test_split(df,random_state=42,test_size=0.2)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("ingestion of data is complated")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)