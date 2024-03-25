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
    # sp500_symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol.to_list()
    context.asset = symbol('AAPL') # TODO: PREDICE PARA TODOS LOS ASSETS QUE FUE ENTRENADO
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0)) # TODO: ver comision MERVAL
    context.set_slippage(slippage.VolumeShareSlippage()) # TODO: ver slippage MERVAL
    context.i = 0


def handle_data(context, data):
    # TODO: usar modelo para predecir precio de stocks. Usar datos del episodio actual para predecir el valor en el siguiente.
    # En base una regla de decision, comprar/vender stocks correspondientes.

    # order_target(ticker, 100) para comprar 100 acciones de un ticker
    # record(key=value) para almacenar datos en el dataframe de resultados

    # en cada ep, evaluar si la prediccion es positiva (con un margen de seguridad) y si es, comprar. Si no, vender.
    context.i += 1
    # price = data.current(context.asset, 'price') solo sirve si quiero ajustar el modelo
    prediction = model.predict(context.i, context.asset.symbol) # modelo se entreno con indices 0 a 730, como uso los indices? tomar en cuenta  online training en cada episodio?
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