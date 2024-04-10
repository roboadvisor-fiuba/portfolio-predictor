from zipline.api import symbol, get_datetime, record
import joblib
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../models'))
from test.analyze_model import analyze_model

model = joblib.load("../models/linear_regression_model.plk")

def initialize(context):
    context.assets = None

def handle_data(context, data):
    date_without_timezone = get_datetime().replace(tzinfo=None)
    predictions = model.predict(date_without_timezone)
    
    if not context.assets:
        context.assets = predictions.keys()
    
    predictions_values = [prediction[0][0] for prediction in predictions.values()] # no se por que la prediccion viene dentro de dos listas
    record(predictions=predictions_values)

    real_values = [data.current(symbol(ticker), 'price') for ticker in predictions.keys()]
    record(values=real_values)

def analyze(context, perf):
    analyze_model(context, perf)