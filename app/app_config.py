import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# app/app_config.py

class AppConfig(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
    HUGGING_FACE_API_KEY: str = os.getenv("HUGGING_FACE_API_KEY", "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")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    FAISS_INDEX_PATH: str = "indexes/faiss_index"
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "rag_db")
    connection_string: str = "mongodb+srv://ruser1896:o9G35A4pHBXip8YF@cluster0.rwrue.mongodb.net/"

    class Config:
        env_file = ".env"  # 👈 optional, load from .env
        
CONFIG = AppConfig()
