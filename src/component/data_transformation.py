import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

from src.exception import CustomException
from src.logger import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object
@dataclass
class DataTransformationconfig:
    current_directory = os.getcwd()
    preprocessor_obj_file = os.path.join(current_directory,"artifacts","preprocessor.pkl")
    logging.info(f"pickle file location -  {preprocessor_obj_file}")

class DataTransformation:
    def __init__(self):
        self.DataTransformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Entered get_data_transformation_object function in the class")
            num_cols = ["writing_score","reading_score"]
            cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                        'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
              steps=[
                      ("imputer",SimpleImputer(strategy="median")),
                      ("scalar", StandardScaler(with_mean=False))
              ]
            )
            
            logging.info("Numerical columns encoding completed")

            cat_pipeline = Pipeline(
                steps=[
                         ("imputer",SimpleImputer(strategy="most_frequent")),
                         ("one_hot_encoder",OneHotEncoder()),
                         ("scalar",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Catagorical columns Encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_cols),
                    ("cat_pipline",cat_pipeline,cat_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df =  pd.read_csv(test_path)

            logging.info("Train and Test data had been read")
            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformation_object()

            target_col_name ="math_score" 
            #num_cols = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessor_obj.transform(input_feature_test_df)
            
            # After preprocessing concatenate column wise with transformed data and targeted data
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                self.DataTransformation_config.preprocessor_obj_file,
                preprocessor_obj
            )

            logging.info("Saved preprocessing object")

            return(
                train_arr,
                test_arr,
                self.DataTransformation_config.preprocessor_obj_file,
            )
        except Exception as e:
            raise CustomException(e,sys)