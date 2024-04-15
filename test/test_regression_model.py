from zipline.api import order_target, symbol, get_datetime
import joblib
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../models'))
from test.analyze_portfolio import analyze_portfolio
from test.workflow import workflow
from functools import partial

model = joblib.load("../models/linear_regression_model.plk")

def initialize(context):
    context.prev_prediction = 0

def signal(context, prediction):
    delta = context.prev_prediction - prediction
    context.prev_prediction = prediction
    return delta < 0 # TODO: agregar umbral

def optimizer(signals):
    def buy(signal):
        return 1000 if signal else 0
    return {k: buy(v) for k, v in signals.items()}

def handle_data(context, data):
    # Get features
    date = get_datetime().replace(tzinfo=None)

    # Run workflow
    signal_partial = partial(signal, context)
    portfolio = workflow(date, predictor=model.predict, signal=signal_partial, optimizer=optimizer)

    # Execute trades
    for ticker, allocation in portfolio.items():
        order_target(symbol(ticker), allocation)

def analyze(context, perf):
    analyze_portfolio(perf)