from pymongo import MongoClient
from app.app_config import CONFIG
# Replace the following with your MongoDB Atlas connection string


# Create a MongoClient instance
client = MongoClient(CONFIG.connection_string)

# Access a specific database
db = client['sample_mflix']

# Access a specific collection
collection = db['final_pages']

class MongoWriter:
    def __init__(self):
        databases = client.list_database_names()
        print("Connection successful. Databases:", databases)
        self.collection = collection

    def write(self, data):
        try:
            data.pop('_id', None)        # Remove any Mongo _id
            data.pop('user_id', None)    # Remove user_id too
            result = self.collection.insert_one(data)
            return result.inserted_id
        except Exception as e:
            print(f"Mongo insert error: {e}")
            return None



    def fetch_all(self):
        """Fetch all documents that are not yet vectorized."""
        try:
            return list(self.collection.find())
        except Exception as e:
            return []
