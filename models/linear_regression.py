'''
Script to train a linear regression model on historical stock data to select the best subset of stocks.
'''

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_historical_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Adj Close']

def calculate_returns(stock_prices):
    return stock_prices.pct_change().dropna()

def perform_linear_regression(returns):
    X = np.arange(len(returns)).reshape(-1, 1)
    y = returns.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0][0]

def select_best_returns(stocks_returns, k):
    selected_tickers = sorted(stocks_returns, key=stocks_returns.get, reverse=True)[:k]
    return selected_tickers

def main():
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    # tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol.to_list()
    start_date = '2020-01-01'
    end_date = '2022-01-01'
    k_best_tickers = 2

    stocks_returns = {}

    for ticker in tickers:
        historical_data = get_historical_data(ticker, start_date, end_date)
        returns = calculate_returns(historical_data)
        slope = perform_linear_regression(returns)
        stocks_returns[ticker] = slope

    print(f"Returns for each stock: {stocks_returns}")
    portfolio = select_best_returns(stocks_returns, k=k_best_tickers)
    print(f"The selected tickers for the portfolio: {portfolio}")

if __name__ == "__main__":
    main()
