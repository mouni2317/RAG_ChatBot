import json
import os

class JsonWriter:
    def __init__(self, file_path="local_db.json"):
        self.file_path = file_path
        # Ensure the file exists
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)

    def write(self, data):
        """Write a new document to the JSON file."""
        try:
            with open(self.file_path, 'r+') as f:
                documents = json.load(f)
                documents.append(data)
                f.seek(0)
                json.dump(documents, f, indent=2)
            return True
        except Exception as e:
            return False

    def fetch_all(self):
        """Fetch all documents that are not yet vectorized."""
        try:
            with open(self.file_path, 'r') as f:
                documents = json.load(f)
                return [doc for doc in documents if not doc.get("is_vectorized", False)]
        except Exception as e:
            return []

    def update_vectorized(self, doc_id):
        """Update a document's vectorized status to True."""
        try:
            with open(self.file_path, 'r+') as f:
                documents = json.load(f)
                for doc in documents:
                    if doc.get("id") == doc_id:
                        doc["is_vectorized"] = True
                        break
                f.seek(0)
                json.dump(documents, f, indent=2)
            return True
        except Exception as e:
            return False 