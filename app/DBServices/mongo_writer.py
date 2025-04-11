from pymongo import MongoClient

class MongoWriter:

    def write(self, data):
        try:
            return self.collection.insert_one(data).inserted_id
        except Exception as e:
            return None
