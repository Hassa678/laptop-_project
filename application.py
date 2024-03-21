from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_LaptopPrice',methods=['POST', 'GET'])
def predict_():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
                 Company = request.form.get('Company'),
                 TypeName = request.form.get('TypeName'),
                 ScreenResolution= request.form.get('ScreenResolution'),
                 Cpu = request.form.get('Cpu'),
                 Memory = request.form.get('Memory'),
                 Gpu = request.form.get('Gpu'),
                 OpSys = request.form.get('OpSys'),
                 Weight = float(request.form.get('Weight')),
                 Ram = float(request.form.get('Ram')),
                 Inches= float(request.form.get('Inches')))
        
        pred_df=data.get_data_as_data_frame()                
        print(pred_df)
        print("Before Prediction")

        predict_pipeline= PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
        
    