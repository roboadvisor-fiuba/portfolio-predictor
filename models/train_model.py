import yfinance as yf
import joblib
from linear_regression import LinearRegressionModel
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)", required=False)
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)", required=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start_date = args.start if args.start else '2014-01-01'
    end_date = args.end if args.end else '2018-01-01'

    stocks_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    model = LinearRegressionModel()

    print("Training starting...")
    model.fit(stocks_data)
    print("Training completed.")

    joblib.dump(model, 'linear_regression_model.plk')
