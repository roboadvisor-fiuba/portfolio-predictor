from flask_restx import Resource, Namespace, fields
from ..models.model_manager import ModelManager

api = Namespace('prediction_model', description='Prediction model operations', path='/')
model_manager = ModelManager()

model_fields = api.model('PredictionModel', {
    'model_type': fields.String(required=True, description='Type of model'),
    'model_file': fields.String(required=True, description='File path to load the model'),
    'input_data': fields.List(fields.Float, required=True, description='Input data for prediction')
})

@api.route('/load')
class LoadModel(Resource):
    @api.expect(model_fields)
    def post(self):
        data = api.payload
        model_type = data['model_type']
        model_file = data['model_file']
        model_manager.load_model(model_type, model_file)
        return {"message": f"Model of type {model_type} loaded successfully"}

@api.route('/predict')
class Predict(Resource):
    @api.expect(model_fields)
    def post(self):
        data = api.payload
        input_data = data['input_data']
        prediction = model_manager.predict(input_data)
        return {"prediction": prediction}