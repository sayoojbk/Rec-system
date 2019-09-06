import time

from ..database import DB

'''
Merchant Data
Example:
{
    “id” : 1,
    “name”: “Food Lion”,
    “location”: “Calle de Zaragoze 4, 28201 Madrid, Madrid”,
    “added_on”: 12342352
    “last_updated”: 12342942,
}
'''
class Merchant(object):

    COLLECTION = "merchants"

    def __init__(self, id, name, location):
        self.id = id;
        self.name = name
        self.location = location
        self.added_on = time.time()
        self.last_updated = self.added_on

    def insert(self):
        if not DB.find_one(Merchant.COLLECTION, {"_id": self.id}):
            DB.insert(collection=Merchant.COLLECTION, data=self.json())

    def json(self):
        return {
            '_id': self.id,
            'name': self.name,
            'location': self.location,
            'added_on': self.added_on,
            'last_updated': self.last_updated
        }
