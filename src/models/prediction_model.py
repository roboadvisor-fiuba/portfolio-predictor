from abc import ABC, abstractmethod
import joblib

class PredictionModel(ABC):
    def __init__(self):
        self.model = None

    def load(self, model_file):
        self.model = joblib.load(model_file)

    def dump(self, model_file):
        joblib.dump(self.model, model_file)

    @abstractmethod
    def predict(self, input_data):
        pass

    @abstractmethod
    def train(self, input_data, labels):
        pass
