from src.exceptions import CustomException
import sys
from src.logger import logging
import pickle
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

# function to store the object
def save_model(model, path):
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(model, file)

    except Exception as e:
        raise CustomException(e, sys)
    
# function to train and evaluate the model
def train_and_eval(x_train, y_train, x_test, y_test, models: dict, params: dict):

    report = {}

    for name, model in models.items():

        logging.info(f"Training model {name}")

        param_dist = params[name]

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            cv=5,
            n_jobs=-1,
            random_state=43,
            refit=True
        )

        search.fit(x_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_

        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)

        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)

        report[name] = {
            "train_score": train_score,
            "test_score": test_score,
            "best_params": best_params
        }

    return report
    
# function to get the best model name
def getBestModelName(report):
    
    best_model = None
    best_score = -float("inf")
    
    for name, scores in report.items():
        
        test_score = scores["test_score"]
        
        if test_score > best_score:
            best_score = test_score
            best_model = name
            
    return best_model

# function to load the object
def load_object(path):
    try:
        logging.info(f"Loading object from file: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        logging.info("Object loaded successfully")

        return obj

    except Exception as e:
        logging.error("Error occurred while loading object")
        raise CustomException(e, sys)