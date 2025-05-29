# db_factory.py
from app_old.DBServices.chroma_writer import ChromaWriter
# from postgres_writer import PostgresWriter  # future expansion
# from dynamodb_writer import DynamoDBWriter  # optional
from app_old.embeddings import embedding_model

def get_db_writer(db_type):
    if db_type == "chroma":
        return ChromaWriter(persist_directory="./chroma_data", embedding_function=embedding_model)
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")
