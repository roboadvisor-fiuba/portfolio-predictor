from .random import RandomModel

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.model_factory = {
            'constant': RandomModel,
            # agregar mas modelos
        }

    def load_model(self, model_type, model_file):
        self.current_model = self.model_factory.get(model_type)
        
        if not self.current_model:
            raise Exception("Invalid model type")

        self.current_model.load(model_file)


    def predict(self, input_data):
        if not self.current_model:
            raise Exception("No model loaded")
        
        return self.current_model.predict(input_data)