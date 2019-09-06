from flask import request
from flask_restplus import Resource
from ..dataModel.category import Category
from ..dataModel.merchant import Merchant
from ..database import DB
from bson.json_util import dumps
import json
from bson.objectid import ObjectId

from ..util.dto import DevEndpoint

api = DevEndpoint.api

@api.route('/v1/category')
@api.doc(params={'id': 'Category ID', 'name': 'Category Name'})
class CategoryAPI(Resource):
    @api.doc('Gets all the categories in the database')
    def get(self):
        """Gets all the categories in the categories collection"""
        categories = DB.find_all("categories")
        return dumps(categories)

    @api.doc('Adds a category to the database')
    def post(self):
        """Adds category to the database."""
        id = request.args.get('id', '')
        name = request.args.get('name', '')
        new_category = Category(id, name)
        new_category.insert()
        return ({'message': 'Successfully Added'}, 200)

@api.route('/v1/merchant')
@api.doc(params={'id': 'Merchant ID', 'name': 'Merchant Name', 'location': 'Merchant Address'})
class MerchantAPI(Resource):
    @api.doc('Gets all the merchants in the database')
    def get(self):
        """Gets all the categories in the categories collection"""
        merchants = DB.find_all("merchants")
        return dumps(merchants)

    @api.doc('Adds a merchant to the database')
    def post(self):
        """Adds merchant to the database."""
        id = request.args.get('id', '')
        name = request.args.get('name', '')
        location = request.args.get('location', '')
        new_merchant = Merchant(id, name, location)
        new_merchant.insert()
        return ({'message': 'Successfully Added'}, 200)


@api.route('/v1/product')
class ProductAPI(Resource):
    @api.doc('Gets all the products in the database')
    def get(self):
        """Gets all the products in the products collection"""
        products = DB.find_all("products")
        return dumps(products)


@api.route('/v1/merchant_category')
class MerchantCategoryAPI(Resource):
    @api.doc('Gets all the merchant categories in the database')
    def get(self):
        """Gets all the products in the products collection"""
        merchant_categories = DB.find_all("merchant_categories")
        return dumps(merchant_categories)


@api.route('/v1/merchant_product')
class MerchantProductAPI(Resource):
    @api.doc('Gets all the merchant products in the database')
    def get(self):
        """Gets all the products in the products collection"""
        merchant_categories = DB.find_all("merchant_products")
        return dumps(merchant_categories)


class Encoder(json.JSONEncoder):
    def default(self):
        if isinstance(obj, ObjectId):
            return str(obj)
        else:
            return obj
