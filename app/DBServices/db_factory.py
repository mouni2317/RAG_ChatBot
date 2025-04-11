# db_factory.py
from app.DBServices.mongo_writer import MongoWriter
# from postgres_writer import PostgresWriter  # future expansion
# from dynamodb_writer import DynamoDBWriter  # optional

def get_db_writer(db_type):
    if db_type == "mongo":
        return MongoWriter()
    # elif db_type == "postgres":
    #     return PostgresWriter()
    # elif db_type == "dynamodb":
    #     return DynamoDBWriter()
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")
