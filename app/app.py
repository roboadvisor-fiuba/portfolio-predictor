from flask import Flask
from flask_restx import Api
from resources.model_resource import LoadModelResource, PredictResource, HealthCheckResource

app = Flask(__name__)
api = Api(app)

api.add_resource(LoadModelResource, '/load_model')
api.add_resource(PredictResource, '/predict')
api.add_resource(HealthCheckResource, '/health')

if __name__ == '__main__':
    app.run(debug=True)
