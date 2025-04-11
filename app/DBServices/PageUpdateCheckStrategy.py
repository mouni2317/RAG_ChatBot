import requests
import hashlib
import sqlite3
from abc import ABC, abstractmethod

# Strategy Interface
class UpdateCheckStrategy(ABC):
    @abstractmethod
    def has_changed(self, url, stored_metadata):
        pass

# Concrete Strategy for HTTP Headers
class HttpHeadersStrategy(UpdateCheckStrategy):
    def has_changed(self, url, stored_metadata):
        response = requests.head(url)
        last_modified = response.headers.get('Last-Modified')
        etag = response.headers.get('ETag')
        return (last_modified != stored_metadata.get('last_modified') or
                etag != stored_metadata.get('etag'))

# Concrete Strategy for Content Hashing
class ContentHashStrategy(UpdateCheckStrategy):
    def has_changed(self, url, stored_metadata):
        response = requests.get(url)
        content_hash = hashlib.sha256(response.text.encode('utf-8')).hexdigest()
        return content_hash != stored_metadata.get('content_hash')

# Context Class
class PageChecker:
    def __init__(self, strategy: UpdateCheckStrategy):
        self.strategy = strategy

    def check_update(self, url, stored_metadata):
        return self.strategy.has_changed(url, stored_metadata)

