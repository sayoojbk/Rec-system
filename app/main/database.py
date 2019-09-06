import pymongo

'''
Creates a database connection and introduces helper functions that call pymongo functions
'''
class DB(object):

    URI = "mongodb://mongodb:27017/"
    DB_NAME = "recommendation"

    @staticmethod
    def init():
        client = pymongo.MongoClient(DB.URI)
        DB.DATABASE = client[DB.DB_NAME]

    @staticmethod
    def insert(collection, data):
        DB.DATABASE[collection].insert(data)

    @staticmethod
    def find_one(collection, query):
        return DB.DATABASE[collection].find_one(query)

    @staticmethod
    def find_all(collection):
        return DB.DATABASE[collection].find()
