from src.exceptions import CustomException
import sys
from src.logger import logging
import pickle
from pathlib import Path

# function to store the object
def save_model(model, path):
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(model, file)

    except Exception as e:
        raise CustomException(e, sys)