from google.cloud import storage
from transformers import AutoModel, AutoTokenizer
import os

class GoogleCloudModelLoader:
    def __init__(self, bucket_name: str, model_path: str):
        """Initialize with Google Cloud Storage bucket and model path"""
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.client = storage.Client()

    def load_embedding_model(self, model_name: str):
        """Download model from Google Cloud and load it"""
        self._download_model(model_name)
        model = AutoModel.from_pretrained(model_name)
        return model

    def load_llm_model(self, model_name: str):
        """Download model from Google Cloud and load it"""
        self._download_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model

    def _download_model(self, model_name: str):
        """Download model files from Google Cloud Storage to a local directory"""
        bucket = self.client.bucket(self.bucket_name)
        model_dir = f"./models/{model_name}"
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        blobs = bucket.list_blobs(prefix=self.model_path)
        for blob in blobs:
            destination_uri = os.path.join(model_dir, blob.name)
            blob.download_to_filename(destination_uri)
