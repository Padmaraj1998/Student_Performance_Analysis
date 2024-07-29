import os
import sys

import pandas as pd
import numpy as np
import dataclasses as dataclass

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from  sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from  src.utils import save_object,evaluate_model
@dataclass
class Model_Trainer_Config:
    current_directory = os.getcwd()
    trained_model_file_path = os.path.join(current_directory,"artifacts","modeltrainer.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_triner_config = Model_Trainer_Config()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and testing data ")
            X_train,Y_train,X_test,Y_test = (
              train_array[:,:-1],
              train_array[:,-1],
              test_array[:,:-1],
              test_array[:,-1]
            )

            models = {
                "Randomforest":RandomForestRegressor(),
                "Adaboost" :AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Xgboost": XGBRegressor(),
                "Catboost": CatBoostRegressor(verbose=False),
                "K-Neighbour": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree":DecisionTreeRegressor()

            }
            parameter = {
                "Decision Tree": {
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Randomforest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "Xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "K-Neighbour":{}
            }
            logging.info("Evaluation model started")
            model_report:dict = evaluate_model(X_train,Y_train,X_test,Y_test,models,parameter)
            logging.info(f"result: {model_report}")
            logging.info("Evaluation model ended")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            print(f"best model name:{best_model_name}")
            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No best model found")
            
            logging.info("Best model found on both testing and training dataset")

            save_object(
                file_path=self.model_triner_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2square = r2_score(Y_test,predicted)
            return r2square
        except Exception as e:
            raise CustomException(e,sys)

        