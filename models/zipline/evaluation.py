'''
Script to evaluate a model using zipline and pyfolio
'''

from zipline.api import order_target, record, symbol
from zipline.finance import commission, slippage
import pandas as pd
import pyfolio as pf
import joblib
import sys        
from os import path
sys.path.insert(0, path.abspath('..')) # needed for joblib loading

model = joblib.load("../linear_regression_model.plk")

# 1. reemplazar stock por todo lo que haya sido entrenado

def initialize(context):
    context.asset = symbol('AAPL') # TODO: test sobre todos los assets en los que fue entrenado el modelo
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0)) # TODO: ver comision MERVAL
    context.set_slippage(slippage.VolumeShareSlippage()) # TODO: ver slippage MERVAL
    context.i = 0

def handle_data(context, data):
    context.i += 1
    prediction = model.predict(context.i, context.asset.symbol) 
    # FIX: modelo se entreno con indices 0 a 730, como usarlo? tomar en cuenta online training en cada episodio?
    action=None
    if prediction > 0:
        order_target(context.asset, 100)
        action = "buy"
    else:
        order_target(context.asset, -100)
        action = "sell"

    record(AAPL=data.current(context.asset, 'price'), action=action, amount=100)


def analyze(context, perf):
    import matplotlib.pyplot as plt
    import numpy as np
    import yfinance as yf

    df = pd.DataFrame(perf)
    pd.DataFrame.to_csv(df, 'result.csv')

    df['portfolio_returns'] = df['portfolio_value'].pct_change()
    portfolio_cumulative_returns = (1 + df['portfolio_returns']).cumprod() - 1

    start_date = '2014-01-01'
    end_date = '2018-01-01'
    sp500_returns = yf.download('SPY', start=start_date, end=end_date)['Adj Close'].pct_change().dropna()
    sp500_cumulative_returns = (1 + sp500_returns).cumprod() - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(df.index.to_numpy(), portfolio_cumulative_returns.to_numpy(), label='Portfolio Returns')
    ax1.plot(sp500_returns.index.to_numpy(), sp500_cumulative_returns.to_numpy(), label='S&P 500 Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns')
    ax1.set_title('Portfolio vs. S&P 500 Cumulative Returns')
    ax1.legend()

    ax2.plot(df.index.to_numpy(), portfolio_cumulative_returns.to_numpy())
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Returns')
    ax2.set_title('Portfolio Cumulative Returns')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
    sharpe_ratio = df['sharpe'].mean()
    portfolio_values = df['portfolio_value'].values
    drawdowns = np.maximum.accumulate(portfolio_values) - portfolio_values
    max_drawdown_percentage = (np.max(drawdowns) / np.max(portfolio_values)) * 100 
    print("Sharpe Ratio:", sharpe_ratio)
    print("Max Drawdown (%):", max_drawdown_percentage)