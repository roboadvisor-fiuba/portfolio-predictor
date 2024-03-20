from flask_restx import Resource, Namespace, fields
import joblib

api = Namespace('prediction_model', description='Prediction model operations', path='/')
current_model = None

load_fields = api.model('PredictionModel', {
    'model_file': fields.String(required=True, description='File path to load the model')
})
predict_fields = api.model('PredictionModel', {
    'input_data': fields.List(fields.Float, required=True, description='Input data for prediction')
})
training_fields = api.model('PredictionModel', {
    'input_data': fields.List(fields.Float, required=True, description='Input data for prediction'),
    'labels': fields.List(fields.Float, required=True, description='Labels for training')
})

@api.route('/load')
class LoadModel(Resource):
    @api.expect(load_fields)
    def post(self):
        global current_model
        data = api.payload
        model_file = data['model_file']
        try:
            current_model = joblib.load(model_file)
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