from pymongo import MongoClient

# Replace the following with your MongoDB Atlas connection string


# Create a MongoClient instance
client = MongoClient(connection_string)

# Access a specific database
db = client['sample_mflix']

# Access a specific collection
collection = db['sessions']

class MongoWriter:
    def __init__(self):
        databases = client.list_database_names()
        print("Connection successful. Databases:", databases)
        self.collection = collection

    def write(self, data):
        try:
            return self.collection.insert_one(data).inserted_id
        except Exception as e:
            return None

    def fetch_all(self):
        """Fetch all documents that are not yet vectorized."""
        try:
            return list(self.collection.find())
        except Exception as e:
            return []
