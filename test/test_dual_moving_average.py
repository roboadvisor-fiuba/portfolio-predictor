from zipline.api import order_target, record, symbol
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from test.analyze_portfolio import analyze_portfolio

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')
    context.short_window = 50
    context.long_window = 200

def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return

    short_mavg = data.history(context.asset, 'price', bar_count=context.short_window, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=context.long_window, frequency="1d").mean()

    if short_mavg > long_mavg:
        order_target(context.asset, 10000)
    elif short_mavg < long_mavg:
        order_target(context.asset, -10000)

    record(AAPL=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)


def analyze(context, perf):
    analyze_portfolio(perf)