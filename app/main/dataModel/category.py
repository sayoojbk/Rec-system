import time

from ..database import DB

'''
Category Data
Example:
{
    “id” : 1,
    “name”: “soda”,
    “added_on”: 12342352
    “last_updated”: 12342942,
}

'''
class Category(object):

    COLLECTION = "categories"

    def __init__(self, id, name):
        self.id = id;
        self.name = name
        self.added_on = time.time()
        self.last_updated = self.added_on

    def insert(self):
        if not DB.find_one(Category.COLLECTION, {"_id": self.id}):
            DB.insert(collection=Category.COLLECTION, data=self.json())

    def json(self):
        return {
            '_id': self.id,
            'name': self.name,
            'added_on': self.added_on,
            'last_updated': self.last_updated,
        }
