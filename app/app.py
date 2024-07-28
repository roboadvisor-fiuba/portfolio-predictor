from flask import Flask
from flask_restx import Api
from resources.predictive_model import api as model_api
from resources.health import HealthCheck
from models import db
from config import Config

app = Flask(__name__)
api = Api(app)
app.config.from_object(Config)
db.init_app(app)
with app.app_context():
    db.create_all()

api.add_namespace(model_api)
api.add_resource(HealthCheck, '/health')

if __name__ == '__main__':
    app.run(debug=True)
