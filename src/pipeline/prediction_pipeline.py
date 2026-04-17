"""
Prediction pipeline

1) receive processed dataframe
2) load preprocessor and model
3) transform data
4) return prediction
"""

import sys
import pandas as pd

from src.logger import logging
from src.exceptions import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig


# class to handle form data
class CustomData:

    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_dataframe(self):
        try:

            logging.info("Creating dataframe from user input")

            data = {
                "gender": self.gender,
                "race_ethnicity": self.race_ethnicity,
                "parental_level_of_education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test_preparation_course": self.test_preparation_course,
                "reading_score": self.reading_score,
                "writing_score": self.writing_score
            }

            df = pd.DataFrame([data])

            logging.info("Dataframe created successfully")

            return df

        except Exception as e:
            raise CustomException(e, sys)


# prediction pipeline class
class PredictPipeline:

    def predict(self, features):

        try:

            logging.info("Starting prediction pipeline")

            # load preprocessor
            preprocessor_path = DataTransformationConfig().preprocessor_model_path
            logging.info(f"Loading preprocessor from {preprocessor_path}")

            preprocessor = load_object(preprocessor_path)

            # load model
            model_path = ModelTrainerConfig().model_object_path
            logging.info(f"Loading model from {model_path}")

            model = load_object(model_path)

            # transform data
            logging.info("Applying preprocessing")

            data_scaled = preprocessor.transform(features)

            # predict
            logging.info("Making prediction")

            prediction = model.predict(data_scaled)

            logging.info(f"Prediction completed: {prediction}")

            return prediction

        except Exception as e:
            logging.error("Error during prediction")
            raise CustomException(e, sys)