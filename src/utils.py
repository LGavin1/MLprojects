import os
import sys
import logging
import pandas as pd 
import numpy as np
import pickle
import dill

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException

os.system('pip install dill')

def save_obej_as_pickle(file_path, obj):
    """
    Save the object as a pickle file
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_path)

        logging.info(f"Object saved as pickle file at {file_path}")

    except Exception as e:
         raise CustomException(e, sys)
