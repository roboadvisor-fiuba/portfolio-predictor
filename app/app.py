from flask import Flask
from flask_restx import Api
from resources.predictive_model import api as model_api
from resources.health import HealthCheck
from models.linear_regression import LinearRegressionModel # por que se necesita este import?

app = Flask(__name__)
api = Api(app)

api.add_namespace(model_api)
api.add_resource(HealthCheck, '/health')

if __name__ == '__main__':
    app.run(debug=True)
