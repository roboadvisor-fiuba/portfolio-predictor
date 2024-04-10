import yfinance as yf
import joblib
from linear_regression import LinearRegressionModel
import argparse

def get_historical_returns(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    stock_prices = stock_data['Adj Close']
    return stock_prices.pct_change().dropna()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process start and end dates.")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)", required=False)
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)", required=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start_date = args.start if args.start else '2014-01-01'
    end_date = args.end if args.end else '2018-01-01'
    
    stocks_data = get_historical_returns(tickers, start_date, end_date)
    model = LinearRegressionModel()

    print("Training starting...")
    model.fit(stocks_data)
    print("Training completed.")

    joblib.dump(model, 'linear_regression_model.plk')
