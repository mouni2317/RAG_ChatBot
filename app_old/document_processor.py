import os
import requests
import asyncio
import json
import csv
import uuid
from fastapi import HTTPException
from io import StringIO

class DocumentProcessor:
    """Handles file processing for different formats (TXT, JSON, CSV)."""

    def __init__(self, path_or_url: str):
        self.path_or_url = path_or_url
        self.text = None
        self.file_type = None

    async def process(self):
        """Determine file type and extract content."""
        if self._is_url():
            await self._fetch_url()
        else:
            await self._read_local_file()

        return {
            "doc_id": str(uuid.uuid4()),  # Generate a unique document ID
            "file_type": self.file_type,
            "content": self.text  # This can be modified as per storage requirements
        }

    def _is_url(self) -> bool:
        """Check if input is a URL."""
        return self.path_or_url.startswith(("http://", "https://"))

    async def _fetch_url(self):
        """Fetch document from a URL."""
        try:
            response = requests.get(self.path_or_url, timeout=10)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch the URL")
            self.text = response.text
            self.file_type = "txt"  # Assuming URL content is plain text
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Request error: {str(e)}")

    async def _read_local_file(self):
        """Read and process a local file."""
        if not os.path.exists(self.path_or_url):
            raise HTTPException(status_code=400, detail="File path does not exist")

        file_extension = self.path_or_url.split('.')[-1].lower()
        raw_text = await asyncio.to_thread(lambda: open(self.path_or_url, "r", encoding="utf-8").read())

        if file_extension == "json":
            self.text = self._parse_json(raw_text)
            self.file_type = "json"
        elif file_extension == "csv":
            self.text = self._parse_csv(raw_text)
            self.file_type = "csv"
        elif file_extension == "txt":
            self.text = raw_text
            self.file_type = "txt"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    def _parse_json(self, raw_text: str):
        """Parse JSON file content."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")

    def _parse_csv(self, raw_text: str):
        """Parse CSV file content into a list of lists."""
        try:
            csv_reader = csv.reader(StringIO(raw_text))
            return [row for row in csv_reader]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid CSV file")
