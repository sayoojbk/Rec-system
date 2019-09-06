import time

from ..database import DB

'''
Product Data
Example:
{
    “_id”: 2,
    “nationality”: 3,
    "age": 22,
    “average_spent”: 24.37,
    “added_on”: 12342352
    “last_updated”: 12342942,
 }



'''
class User(object):

    COLLECTION = "user"

    def __init__(self, user_id, age, nationality, average_spent):
        self.id = user_id
        self.age = age
        self.user_id = user_id
        self.nationality = nationality
        self.average_spent = average_spent
        self.added_on = time.time()
        self.last_updated = self.added_on

    def insert(self):
        if not DB.find_one(User.COLLECTION, {"_id": self.id}):
            DB.insert(collection=User.COLLECTION, data=self.json())

    def json(self):
        return {
            '_id': self.id,
            'nationality': self.nationality,
            'age': self.age,
            'average_spent': self.average_spent,
            'purchased_count': self.categories,
            'added_on': self.added_on,
            'last_updated': self.last_updated
        }
