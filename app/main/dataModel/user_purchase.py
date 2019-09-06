import time

from ..database import DB

'''
Product Data
Example:
{
    “id”: 1,
    “user_id”: 2,
    “product_id”: 3,
    “purchased_count”: 4
    "added_on”: 12342352”,
    “last_updated”: 12342942
 }


'''
class UserPurchase(object):

    COLLECTION = "user_purchases"

    def __init__(self, id, user_id, product_id, purchased_count):
        self.id = id
        self.user_id = user_id
        self.product_id = product_id
        self.purchased_count = purchased_count
        self.added_on = time.time()
        self.last_updated = self.added_on

    def insert(self):
        if not DB.find_one(UserPurchase.COLLECTION, {"_id": self.id}):
            DB.insert(collection=UserPurchase.COLLECTION, data=self.json())

    def json(self):
        return {
            '_id': self.id,
            'user_id': self.name,
            'product_id': self.retail_price,
            'purchased_count': self.categories,
            'added_on': self.added_on,
            'last_updated': self.last_updated
        }
