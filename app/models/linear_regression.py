'''
Script to train a linear regression model on historical stock data to select the best subset of stocks.
'''

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def get_historical_returns(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    stock_prices = stock_data['Adj Close']
    return stock_prices.pct_change().dropna()

def select_best_returns(stocks_returns, k):
    selected_tickers = sorted(stocks_returns, key=stocks_returns.get, reverse=True)[:k]
    return selected_tickers

class LinearRegressionModel:
    def __init__(self):
        self.trained = False

    def fit(self, stocks_data):
        models = {}
        self.start_date = stocks_data.index[0]

        for ticker, returns in stocks_data.items():
            model = self.perform_linear_regression(returns)
            models[ticker] = model

        self.models = models
        self.trained = True
        return [model.coef_[0][0] for model in models.values()]

    def predict(self, d):
        assert self.trained, "Model must be trained before making predictions."

        predictions = {}
        X = (pd.Timestamp(d) - self.start_date).days
        for stock, model in self.models.items():
            predictions[stock] = model.predict([[X]])
        
        return predictions
    
    def perform_linear_regression(self, returns):
        X = np.arange(len(returns)).reshape(-1, 1)
        y = returns.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)

        return model

def main():
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start_date = '2020-01-01'
    end_date = '2022-01-01'
    predict_date = '2022-01-02'
    k_best_tickers = 2

    stocks_data = get_historical_returns(tickers, start_date, end_date)
    model = LinearRegressionModel()
    slopes = model.fit(stocks_data)
    stocks_predictions = model.predict(predict_date)

    print(f"Slopes of each linear regression: ", slopes)
    print(f"Predicted returns for each stock at day {predict_date}: {stocks_predictions}")
    portfolio = select_best_returns(stocks_predictions, k=k_best_tickers)
    print(f"The selected tickers for the portfolio: {portfolio}")

    joblib.dump(model, 'linear_regression_model.plk')

if __name__ == "__main__":
    main()
