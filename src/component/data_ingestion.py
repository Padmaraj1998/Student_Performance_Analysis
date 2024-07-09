#Geting the data from other sources
import os
import sys
from src.exception import CustomException 
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationconfig

@dataclass
class DataIngestionConfig:
    cur_direct = os.getcwd()
    train_data_path: str=os.path.join(cur_direct,'artifacts','train.csv')
    test_data_path : str=os.path.join(cur_direct,'artifacts','test.csv')
    raw_data_path: str=os.path.join(cur_direct,'artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info(f"Raw data file had been created")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Raw data had been splited")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Train data had been created")
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Test data had been created")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
 
        except Exception as e:
            logging.info(f"Exception has occurred,{e}")

if __name__=="__main__":    
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    transformed_train_data, transformed_test_data, preprocessor_obj_file = data_transformation.initiate_data_transformation(train_data,test_data)