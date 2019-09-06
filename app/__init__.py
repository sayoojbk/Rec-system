from flask_restplus import Api
from flask import Blueprint

from .main.controller.health_controller import api2 as health_ns
from .main.controller.dev_controller import api as dev_ns


blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='FLASK RESTPLUS API BOILER-PLATE WITH JWT',
          version='1.0',
          description='a boilerplate for flask restplus web service'
          )

api.add_namespace(health_ns, path='/health')
api.add_namespace(dev_ns, path='/dev')
