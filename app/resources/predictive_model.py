from flask_restx import Resource, Namespace, fields
import joblib
import numpy as np
import models
from datetime import datetime
import yfinance as yf

api = Namespace('predictive_model', description='Predictive model operations', path='/')
current_model = None

load_fields = api.model('PredictiveModel', {
    'path': fields.String(required=True, description='Path to load the model file')
})
predict_fields = api.model('PredictiveModel', {
    'date': fields.List(fields.Float, required=True, description='Date for prediction')
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
            current_model = joblib.load(model_file) # NOTE: path is relative to the os.cwd(), where the app was executed
        except FileNotFoundError:
            return {"message": "File not found"}, 400
        return {"message": "Model loaded successfully"}

@api.route('/predict')
class Predict(Resource):
    @api.expect(predict_fields)
    def post(self):
        global current_model
        data = api.payload
        date = data['date'] # DATE MUST COME IN FORMAT yyyy-mm
        if not current_model:
            return {"message": "No model loaded"}, 400

        db_prediction = models.Prediction.query.filter_by(month=date).first
        if not db_prediction:
            return {"prediction": db_prediction.prediction}

        prediction = current_model.predict(date)
        # TODO: apply strategy to the model output
        # The prediction should be an array with (ticker, fraction)

        new_prediction = models.Prediction(month=date, prediction=prediction)
        models.db.session.add(new_prediction)
        models.db.session.commit()
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
        # TODO: validate if the loaded model accepts partial fitting
        current_model.partial_fit(input_data, labels)
        return {"message": "Training completed successfully"}

@api.route('/stats')
class Stats(Resource):
    @api.expect()
    def get(self):
        global current_model

        historical_predictions = models.Prediction.query.all()

        sorted_predictions = sorted(historical_predictions, key=lambda x: datetime.strptime(x.month, "%Y-%m"))
        if len(sorted_predictions) <= 1:
            return {'index': [], 'portfolio': [], 'dates': []}
        dates = [datetime.strptime(pred.month, "%Y-%m") for pred in sorted_predictions]
        
        unique_tickers = set()
        unique_tickers.add("SPY") # FIX: indice merval no tiene todos los datos en yfinance. Sacarlos de otra api. Por ahora, se usa SP500
        for pred in sorted_predictions:
            for (ticker, _) in pred.prediction:
                unique_tickers.add(ticker) 

        def get_next_month(date_str):
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date.month == 12:
                next_month = date.replace(year=date.year + 1, month=1, day=1)
            else:
                next_month = date.replace(month=date.month + 1, day=1)
            return next_month.strftime("%Y-%m-%d")

        prices = yf.download(unique_tickers, start=sorted_predictions[0].month+'-01', end=get_next_month(sorted_predictions[-1].month+'-01'), progress=False, interval='1mo')
        value_merval = [prices.loc[prices.index == date, ('Close', 'SPY')].iloc[-1] for date in dates]
        value_predictions = [sum([((prices.loc[prices.index == datetime.strptime(pred.month, "%Y-%m"), ('Close', ticker)].iloc[-1]) * fraction) for (ticker, fraction) in pred.prediction]) for pred in sorted_predictions]

        return {'index_value': value_merval, 'portfolio_value': value_predictions, 'dates': [pred.month for pred in sorted_predictions]}