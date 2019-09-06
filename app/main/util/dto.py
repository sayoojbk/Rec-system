from flask_restplus import Namespace, fields

class Health:
    api = Namespace('health', description='Checks Recommendation Service Health')

class DevEndpoint:
    api = Namespace('dev', description='Used for testing, wont be publicly available')
