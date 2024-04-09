from zipline.api import order_target, symbol, get_datetime, record
import joblib
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../models'))
from test.analyze_result import analyze_result

model = joblib.load("../models/linear_regression_model.plk")

def initialize(context):
    context.assets = None

def handle_data(context, data):
    date_without_timezone = get_datetime().replace(tzinfo=None)
    predictions = model.predict(date_without_timezone)
    
    if not context.assets:
        context.assets = predictions.keys()
    
    predictions_values = [prediction[0][0] for prediction in predictions.values()] # no se por que la prediccion viene dentro de dos listas
    record(predictions=predictions_values)

    real_values = [data.current(symbol(ticker), 'price') for ticker in predictions.keys()]
    record(values=real_values)

def analyze(context, perf):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = pd.DataFrame(perf)
    pd.DataFrame.to_csv(df[['predictions', 'values']], 'result.csv')

    for i, asset in enumerate(context.assets):
        asset_predictions = np.array([p[i] for p in df['predictions']])
        asset_real_values = np.array([p[i] for p in df['values']])

        predicted_values = asset_predictions.cumsum() + asset_real_values[0]

        mae = mean_absolute_error(asset_real_values, predicted_values)
        mse = mean_squared_error(asset_real_values, predicted_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((asset_real_values - predicted_values) / asset_real_values)) * 100
        r2 = r2_score(asset_real_values, predicted_values)
        
        print('Results for asset ', asset)

        print("\tMean Absolute Error (MAE):", mae)
        print("\tMean Squared Error (MSE):", mse)
        print("\tRoot Mean Squared Error (RMSE):", rmse)
        print("\tMean Absolute Percentage Error (MAPE):", mape)
        print("\tR-squared (R2):", r2)

        plt.figure(figsize=(10, 5))
        plt.plot(df.index.to_numpy(), asset_real_values, label='Value', color='blue')
        plt.plot(df.index.to_numpy(), predicted_values, label='Prediction', linestyle='--', color='red')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Predictions vs. Real Values: ' + asset)
        plt.legend()
        plt.grid(True)
        plt.show()

