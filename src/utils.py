import os
import sys
import logging
import pandas as pd 
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score


from src.exception import CustomException

os.system('pip install dill')

def save_object(file_path, obj):
    """
    Save the object as a pickle file
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved as pickle file at {file_path}")

    except Exception as e:
         raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the models and return the best model based on R2 score
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            report[model_name] = r2
        return report

    except Exception as e:
        raise CustomException(e, sys)