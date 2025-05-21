import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Data Transformation Configuration
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Data Transformation Class
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Get the data transformer object for preprocessing
        This is the main function that creates the preprocessing pipeline for numerical and categorical features.
        It uses the ColumnTransformer to apply different transformations to different columns.
        """
        try:
            numerical_features = ['writing score', 'reading score']
            # Categorical features
            categorical_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=
                    [
                        ("imputer", SimpleImputer(strategy='most_frequent')),
                        ("onehot", OneHotEncoder(handle_unknown='ignore')),
                        ("scaler", StandardScaler(with_mean=False))
                    ]
            )
            
            # Log the features being used for transformation
            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")
          
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )
            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            
            logging.info("Train and Test data read successfully")

            logging.info("Obtaining preprocessing object")
            
            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'math score'
            numerical_columns = ['writing score', 'reading score']
            
            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            target_features_test_df = test_df[target_column]

            logging.info(
                f"Applying preprocessing object on training and testing dataframes"
                )
            
            input_features_train_arr= preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr= preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
                ]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df)]

            logging.info(f"Preprocessing object completed")

            # Save the preprocessor object to a file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )





        except Exception as e:
            raise CustomException(e, sys)