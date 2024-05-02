from zipline.api import order_target_percent, symbol, get_datetime
from zipline import run_algorithm
import joblib
import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../models'))
from test.analyze_portfolio import analyze_portfolio
from test.workflow import workflow
from functools import partial

model = joblib.load("../models/linear_regression_model.plk")

def initialize(context):
    context.prev_prediction = 0

def signal(context, prediction):
    delta = prediction - context.prev_prediction
    context.prev_prediction = prediction
    return delta

def selector(signals):
    threshold = 0
    return {k: v > threshold for k, v in signals.items()}

def optimizer(selection):
    num_assets = sum(1 for value in selection.values() if value)
    target_weight = 1.0 / num_assets if num_assets else 0
    return {k: target_weight if v else 0 for k, v in selection.items()}

def handle_data(context, data):
    # Get features
    date = get_datetime().replace(tzinfo=None)

    # Run workflow
    signal_partial = partial(signal, context)
    portfolio = workflow(date, predictor=model.predict, signal=signal_partial, optimizer=optimizer)

    # Execute trades    
    for ticker, percentage in portfolio.items():
        order_target_percent(symbol(ticker), percentage)

def analyze(context, perf):
    analyze_portfolio(perf)

if __name__ == '__main__':
    start_date = pd.Timestamp(2014, 1, 1)
    end_date = pd.Timestamp(2018, 1, 1)
    capital_base = 1000000.0

    results = run_algorithm(
        start=start_date,
        end=end_date,
        initialize=initialize,
        capital_base=capital_base,
        handle_data=handle_data,
        analyze=analyze,
        bundle='quandl'
    )
