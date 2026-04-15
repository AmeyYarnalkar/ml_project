import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import sys

from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_model, train_and_eval, getBestModelName
from src.config import model_config

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    base = Path.cwd()
    model_object_path = base / "artifacts" / "model.pkl"


class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()

    def train_model(self, train_arr, test_arr):

        try:

            logging.info("Splitting training and test arrays")

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            logging.info("Initializing regression models")

            models = model_config.MODELS
            
            params = model_config.PARAMS
            
            logging.info("Training and evaluating models")

            report = train_and_eval(x_train, y_train, x_test, y_test, models, params)

            logging.info(f"Model evaluation report: {report}")

            best_model_name = getBestModelName(report)

            logging.info(f"Best model selected: {best_model_name}")

            best_model = models[best_model_name]

            best_params = report[best_model_name]["best_params"]
            best_model.set_params(**best_params)

            logging.info("Training best model on full training data")

            best_model.fit(x_train, y_train)

            y_pred = best_model.predict(x_test)

            score = r2_score(y_test, y_pred)

            logging.info(f"Best model R2 score on test data: {score}")

            logging.info("Saving trained model")

            self.config.model_object_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            save_model(
                path=self.config.model_object_path,
                model=best_model
            )

            logging.info("Model saved successfully")

            return score

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)