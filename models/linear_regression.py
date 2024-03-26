import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from models.predictive_model_interface import PredictiveModelInterface

class LinearRegressionModel(PredictiveModelInterface):
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

    def predict(self, d, ticker=None):
        assert self.trained, "Model must be trained before making predictions."

        predictions = {}
        X = (pd.Timestamp(d) - self.start_date).days

        if ticker:
            return self.models[ticker].predict([[X]])

        for ticker, model in self.models.items():
            predictions[ticker] = model.predict([[X]])
        
        return predictions
    
    def partial_fit(self, new_data, new_target):
        pass

    def perform_linear_regression(self, returns):
        X = np.arange(len(returns)).reshape(-1, 1)
        y = returns.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)

        return model