import sys
import os
import pandas as pd

# Assuming exception, logger, and utils are in the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from exception import CustomException
from logger import logging  
from utils import load_object  

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 Company: str,
                 TypeName: str,
                 ScreenResolution: str,
                 Cpu: str,
                 Memory: str,
                 Gpu: str,
                 OpSys: str,
                 Weight: int,
                 Ram: int,
                 Inches: int):
        self.Company = Company
        self.TypeName = TypeName
        self.ScreenResolution = ScreenResolution
        self.Cpu = Cpu
        self.Memory = Memory
        self.Gpu = Gpu
        self.OpSys = OpSys
        self.Weight = Weight
        self.Ram = Ram
        self.Inches = Inches

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Company": [self.Company],
                "TypeName": [self.TypeName],
                "ScreenResolution": [self.ScreenResolution],
                "Cpu": [self.Cpu],
                "Memory": [self.Memory],
                "Gpu": [self.Gpu],
                "OpSys": [self.OpSys],
                "Weight": [self.Weight],
                "Ram": [self.Ram],
                "Inches": [self.Inches],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
