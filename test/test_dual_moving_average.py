from zipline.api import order_target, record, symbol
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
    return short_mavg > long_mavg

def optimizer(signals):
    def buy(signal):
        return 1000 if signal else 0
    return {k: buy(v) for k, v in signals.items()}

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
    for ticker, allocation in portfolio.items():
        order_target(symbol(ticker), allocation)

def analyze(context, perf):
    analyze_portfolio(perf)