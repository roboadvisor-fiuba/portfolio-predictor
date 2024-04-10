from zipline.api import order_target, symbol, get_datetime
import joblib
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../models'))
from test.analyze_result import analyze_result

model = joblib.load("../models/linear_regression_model.plk")

def initialize(context):
    pass

def handle_data(context, data):
    date_without_timezone = get_datetime().replace(tzinfo=None)
    predictions = model.predict(date_without_timezone)

    for ticker, prediction in predictions.items():
        asset = symbol(ticker)
        # TODO: separar estrategia y definir cuánto se compra y vende
        if prediction > 0:
            order_target(asset, 10000)
        else:
            order_target(asset, -10000)

def analyze(context, perf):
    analyze_result(perf)