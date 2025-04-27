# db_factory.py
from app.DBServices.mongo_writer import MongoWriter
from app.DBServices.neo4j_writer import Neo4jWriter
from app.DBServices.faiss_writer import FaissWriter
from app.DBServices.chroma_writer import ChromaWriter
# from postgres_writer import PostgresWriter  # future expansion
# from dynamodb_writer import DynamoDBWriter  # optional
from app.embeddings import embedding_model

def get_db_writer(db_type):
    if db_type == "mongo":
        return MongoWriter()
    elif db_type == "faiss":
        if dimension is None:
            raise ValueError("Dimension must be specified for FaissWriter.")
        return FaissWriter(dimension)
    elif db_type == "neo4j":
        return Neo4jWriter()
    elif db_type == "chroma":
        return ChromaWriter(persist_directory="./chroma_data", embedding_function=embedding_model)
    else:
        raise ValueError(f"Unsupported DB type: {db_type}")
