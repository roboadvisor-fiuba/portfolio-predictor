from .prediction_model import PredictionModel

class RandomModel(PredictionModel):
    def predict(self, input_data):
        return self.model.predict(input_data)
