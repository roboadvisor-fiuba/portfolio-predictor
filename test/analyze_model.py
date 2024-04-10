import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def analyze_model(context, perf):
    df = pd.DataFrame(perf)
    pd.DataFrame.to_csv(df[['predictions', 'values']], 'result.csv')

    for i, asset in enumerate(context.assets):
        predicted_values = np.array([p[i] for p in df['predictions']])
        real_values = np.array([p[i] for p in df['values']])

        mae = mean_absolute_error(real_values, predicted_values)
        mse = mean_squared_error(real_values, predicted_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((real_values - predicted_values) / real_values)) * 100
        r2 = r2_score(real_values, predicted_values)
        
        print('Results for asset ', asset)
        print("\tMean Absolute Error (MAE):", mae)
        print("\tMean Squared Error (MSE):", mse)
        print("\tRoot Mean Squared Error (RMSE):", rmse)
        print("\tMean Absolute Percentage Error (MAPE):", mape)
        print("\tR-squared (R2):", r2)

        plt.figure(figsize=(10, 5))
        plt.plot(df.index.to_numpy(), real_values, label='Value', color='blue')
        plt.plot(df.index.to_numpy(), predicted_values, label='Prediction', linestyle='--', color='red')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Predictions vs. Real Values: ' + asset)
        plt.legend()
        plt.grid(True)
    plt.show()