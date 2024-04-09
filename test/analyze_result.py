import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd

def cumulative_returns(prices):
    return (1 + prices.pct_change()).cumprod() - 1

def analyze_result(perf):
    df = pd.DataFrame(perf)
    pd.DataFrame.to_csv(df, 'result.csv')


    start_date = '2014-01-01'
    end_date = '2018-01-01'
    sp500_values = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
    
    portfolio_cumulative_returns = cumulative_returns( df['portfolio_value'])
    sp500_cumulative_returns = cumulative_returns(sp500_values)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(df.index.to_numpy(), portfolio_cumulative_returns.to_numpy(), label='Portfolio Returns')
    ax1.plot(sp500_values.index.to_numpy(), sp500_cumulative_returns.to_numpy(), label='S&P 500 Returns')
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