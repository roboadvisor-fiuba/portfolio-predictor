import yfinance as yf
import joblib
import sys        
from os import path
sys.path.insert(0, path.abspath('..')) # needed for import linear_regression
from models.linear_regression import LinearRegressionModel

def get_historical_returns(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    stock_prices = stock_data['Adj Close']
    return stock_prices.pct_change().dropna()

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start_date = '2020-01-01'
    end_date = '2022-01-01'
    stocks_data = get_historical_returns(tickers, start_date, end_date)
    model = LinearRegressionModel()

    print("Training starting...")
    model.fit(stocks_data)
    print("Training completed.")

    joblib.dump(model, 'linear_regression_model.plk')
