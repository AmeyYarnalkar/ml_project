"""
Tasks:

1. Handling the logic for the actual data transformation
2. then actually transforming the data
"""

import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_model


@dataclass
class DataTransformationConfig:
    base = Path().cwd()
    preprocessor_model_path = base / "artifacts" / "preprocessor.pkl"


class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    # create preprocessor
    def get_preprocessor(self, num_cols, cat_cols):

        try:
            logging.info("Creating preprocessing pipelines for numerical and categorical features")

            oh_encoder = OneHotEncoder(handle_unknown="ignore")
            scaler = StandardScaler()

            preprocessor = ColumnTransformer([
                ("num", scaler, num_cols),
                ("cat", oh_encoder, cat_cols)
            ])

            logging.info("Preprocessor object created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # actual transformation
    def transform_data(self, train_path, test_path):

        try:
            logging.info("Starting data transformation process")

            # read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded")

            target_col_name = "math_score"

            input_feature_train = train_df.drop(target_col_name, axis=1)
            input_feature_test = test_df.drop(target_col_name, axis=1)

            target_feature_train = train_df[target_col_name]
            target_feature_test = test_df[target_col_name]

            logging.info("Input features and target variable separated")

            num_cols = input_feature_train.select_dtypes(exclude="object").columns
            cat_cols = input_feature_train.select_dtypes(include="object").columns

            logging.info("Identified numerical and categorical columns")

            preprocessor = self.get_preprocessor(num_cols, cat_cols)

            logging.info("Applying preprocessing on training data")

            input_feature_train = preprocessor.fit_transform(input_feature_train)

            logging.info("Applying preprocessing on test data")

            input_feature_test = preprocessor.transform(input_feature_test)

            logging.info("Saving the fitted preprocessor object")

            save_model(preprocessor, self.config.preprocessor_model_path)

            train_arr = np.c_[input_feature_train, target_feature_train]
            test_arr = np.c_[input_feature_test, target_feature_test]

            logging.info("Data transformation completed successfully")

            return train_arr, test_arr, self.config.preprocessor_model_path

        except Exception as e:
            raise CustomException(e, sys)