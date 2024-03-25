from flask_restx import Resource, Namespace, fields
import joblib
import numpy as np

api = Namespace('predictve_model', description='Predictive model operations', path='/')
current_model = None

load_fields = api.model('PredictiveModel', {
    'path': fields.String(required=True, description='Path to load the model file')
})
predict_fields = api.model('PredictiveModel', {
    'input_data': fields.List(fields.Float, required=True, description='Input data for prediction')
})
training_fields = api.model('PredictiveModel', {
    'input_data': fields.List(fields.Float, required=True, description='Input data for prediction'),
    'labels': fields.List(fields.Float, required=True, description='Labels for training')
})

@api.route('/load')
class LoadModel(Resource):
    @api.expect(load_fields)
    def post(self):
        global current_model
        data = api.payload
        model_file = data['path']
        try:
            current_model = joblib.load(model_file) # EL PATH DEPENDE DE DONDE SE HAYA EJECUTADO APP.PY (os.getcwd())
        except FileNotFoundError:
            return {"message": "File not found"}, 400
        return {"message": "Model loaded successfully"}

@api.route('/predict')
class Predict(Resource):
    @api.expect(predict_fields)
    def post(self):
        global current_model
        data = api.payload
        input_data = data['input_data']
        if not current_model:
            return {"message": "No model loaded"}, 400
        prediction = current_model.predict(input_data)

        for key, value in prediction.items():
            if isinstance(value, np.ndarray):
                prediction[key] = value.tolist()

        return {"prediction": prediction}

@api.route('/partial_fit')
class Train(Resource):
    @api.expect(training_fields)
    def post(self):
        data = api.payload
        input_data = data['input_data']
        labels = data['labels']
        if not current_model:
            return {"message": "No model loaded"}, 400
        current_model.partial_fit(input_data, labels)
        return {"message": "Training completed successfully"}