from flask_restx import Resource, Namespace

api = Namespace('health', description='Health check')

@api.route('/health')
class HealthCheck(Resource):
    def get(self):
        return {"status": "ok"}
