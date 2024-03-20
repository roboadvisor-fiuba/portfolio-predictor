'''
Script to evaluate a dual_moving_average strategy using zipline
Run with: 'zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark'
'''

from zipline.api import order_target, record, symbol
from zipline.finance import commission, slippage
import pandas as pd

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')
    context.short_window = 50
    context.long_window = 200
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())

def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return

    short_mavg = data.history(context.asset, 'price', bar_count=context.short_window, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=context.long_window, frequency="1d").mean()

    if short_mavg > long_mavg:
        order_target(context.asset, 100)
    elif short_mavg < long_mavg:
        order_target(context.asset, 0)

    record(AAPL=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)


def analyze(context, perf):
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.DataFrame(perf)
    start_date = '2014-01-01'
    end_date = '2018-01-01'

    df['portfolio_returns'] = df['portfolio_value'].pct_change()
    portfolio_cumulative_returns = (1 + df['portfolio_returns']).cumprod() - 1

    sp500_returns = get_sp500(start_date, end_date)
    sp500_cumulative_returns = (1 + sp500_returns).cumprod() - 1
    
    sharpe_ratio = df['sharpe'].mean()
    portfolio_values = df['portfolio_value'].values
    drawdowns = np.maximum.accumulate(portfolio_values) - portfolio_values
    max_drawdown = np.max(drawdowns)  # Absolute value of maximum drawdown
    peak_value = np.max(portfolio_values)
    max_drawdown_percentage = (max_drawdown / peak_value) * 100  # Convert to percentage

    plt.figure(figsize=(12, 6))
    plt.plot(df.index.to_numpy(), portfolio_cumulative_returns.to_numpy(), label='Portfolio Returns')
    plt.plot(sp500_returns.index.to_numpy(), sp500_cumulative_returns.to_numpy(), label='S&P 500 Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Portfolio vs. S&P 500 Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Sharpe Ratio:", sharpe_ratio)
    print("Max Drawdown (%):", max_drawdown_percentage)


def get_sp500(start_date, end_date):
    import yfinance as yf
    
    return yf.download('SPY', start=start_date, end=end_date)['Adj Close'].pct_change().dropna()