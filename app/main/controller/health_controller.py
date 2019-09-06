from flask import request
from flask_restplus import Resource

from ..util.dto import Health

api2 = Health.api

@api2.route('/v1/marco')
class HealthCheck(Resource):
    @api2.doc('Checks Health Of Service')
    def get(self):
        """Checks Health of Service"""
        return {'statusCode': 1, 'message': 'recommendation'}

