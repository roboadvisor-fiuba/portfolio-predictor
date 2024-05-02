from zipline import run_algorithm
from zipline.api import order_target_percent, symbol
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from test.analyze_portfolio import analyze_portfolio
from test.workflow import workflow

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')
    context.short_window = 50
    context.long_window = 200

def signal(prediction):
    short_mavg, long_mavg = prediction
    return short_mavg - long_mavg

def selector(signals):
    threshold = 0
    return {k: v > threshold for k, v in signals.items()}
    
def optimizer(selection):
    num_assets = sum(1 for value in selection.values() if value)
    target_weight = 1.0 / num_assets if num_assets else 0
    return {k: target_weight if v else 0 for k, v in selection.items()}

def handle_data(context, data):
    context.i += 1
    if context.i < context.long_window:
        return

    # Get features
    short_mavg = data.history(context.asset, 'price', bar_count=context.short_window, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=context.long_window, frequency="1d").mean()
    features = {'AAPL': ((short_mavg, long_mavg))}

    # Run workflow
    portfolio = workflow(features, signal=signal, optimizer=optimizer)

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
