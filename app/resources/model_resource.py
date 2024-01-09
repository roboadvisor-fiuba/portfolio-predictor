from flask_restx import Resource, Namespace, fields
from app.models.model_manager import ModelManager

api = Namespace('model', description='Operaciones con modelos')
model_manager = ModelManager()

model_resource_fields = api.model('ModelResource', {
    'model_type': fields.String(required=True, description='Tipo de modelo'),
    'model_file': fields.String(required=True, description='Ruta del archivo del modelo'),
    'input_data': fields.List(fields.Float, required=True, description='Datos de entrada para predicción')
})

@api.route('/load')
class LoadModelResource(Resource):
    @api.expect(model_resource_fields)
    def post(self):
        data = api.payload
        model_type = data['model_type']
        model_file = data['model_file']
        model_manager.load_model(model_type, model_file)
        return {"message": f"Model of type {model_type} loaded successfully"}

@api.route('/predict')
class PredictResource(Resource):
    @api.expect(model_resource_fields)
    def post(self):
        data = api.payload
        input_data = data['input_data']
        prediction = model_manager.predict(input_data)
        return {"prediction": prediction}

@api.route('/health')
class HealthCheckResource(Resource):
    def get(self):
        return {"status": "Service is running"}
