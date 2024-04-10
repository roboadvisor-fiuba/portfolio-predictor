from zipline.api import order_target, symbol, get_datetime
import joblib
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../models'))
from test.analyze_portfolio import analyze_portfolio

model = joblib.load("../models/linear_regression_model.plk")

def initialize(context):
    context.previous_prediction = 0

def handle_data(context, data):
    date_without_timezone = get_datetime().replace(tzinfo=None)
    predictions = model.predict(date_without_timezone)

    for ticker, prediction in predictions.items():
        asset = symbol(ticker)

        if prediction - context.previous_prediction > 0: # TODO: agregar un umbral de seguridad
            order_target(asset, 10000) # TODO: definir cuanto comprar y vender, estrategia de long/short
        else:
            order_target(asset, -10000)

def analyze(context, perf):
    analyze_portfolio(perf)