from linear_model import LinearModel
from neural_network_model import NeuralNetworkModel

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.model_factory = {
            'linear': LinearModel,
            'neural_network': NeuralNetworkModel,
        }


    def load_model(self, model_type, model_file):
        self.current_model = self.model_factory.get(model_type)
        
        if not self.current_model:
            raise Exception("Invalid model type")

        self.current_model.load(model_file)


    def predict(self, input_data):
        if self.current_model:
            return self.current_model.predict(input_data)
        else:
            raise Exception("No model loaded")