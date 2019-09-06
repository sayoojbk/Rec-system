import time

from ..database import DB

'''
Merchant Category Data
Example:
 {
    “id”: 1,
    “mechant_id”: 245,
    “category_id”: 452,
    “priceLevel”: 1,
    “added_on”: 12342352,
    “last_updated”: 12342942
 }
'''
class Merchant_Category(object):

    COLLECTION = "merchant_categories"

    def __init__(self, id, merchant_id, category_id, price_level):
        self.id = id
        self.merchant_id = merchant_id
        self.category_id = category_id
        self.price_level = price_level
        self.added_on = time.time()
        self.last_updated = self.added_on

    def insert(self):
        if not DB.find_one(Merchant_Category.COLLECTION, {"_id": self.id}):
            DB.insert(collection=Merchant_Category.COLLECTION, data=self.json())

    def json(self):
        return {
            '_id': self.id,
            'merchant_id': self.merchant_id,
            'category_id': self.category_id,
            'price_level': self.price_level,
            'added_on': self.added_on,
            'last_updated': self.last_updated
        }
