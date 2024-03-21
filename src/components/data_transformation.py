import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(parent_dir)


from exception import CustomException
from logger import logging
from utils import save_objects


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifact','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try : 
            numerical_columns = ['Ram', 'Inches', 'Weight']                       
            logging.info("num columns is transformed")
            num_pipeline = Pipeline(steps=[   
            ('impute', SimpleImputer(strategy='median')),
            ("scaler",StandardScaler(with_mean=False))
            ])  
                
            categorical_columns = ['Company', 'TypeName', 'ScreenResolution', 'Cpu',
                           'Memory', 'Gpu', 'OpSys']
            cat_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
            ("scaler", StandardScaler(with_mean=False))
            ])
            logging.info(f'Categorical columns encoding completed: {categorical_columns}')
            preprocessor = ColumnTransformer(transformers=[
                ('cat_pipeline', cat_pipeline, categorical_columns),
                ('num_pipeline', num_pipeline, numerical_columns)
                ])
            return preprocessor

        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data successfully')
            logging.info('obtainnig preprocessing objesct')
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = "Price"
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Aplluying preprocesssing object in test_df and train_df")
            logging.info("Numerical columns transformed")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"saved preprocessing object.")
            
            save_objects(
                file_path = self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessor_obj
            )
            
            
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
                
            )
        except Exception as e:
            raise CustomException(e,sys)
            
            
            
            
            

    