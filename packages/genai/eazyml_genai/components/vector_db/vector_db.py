from abc import ABC
from typing import Dict, Any
from enum import Enum


class VectorDBType(Enum):
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


class VectorDB(ABC):

    def __init__(self, type: VectorDBType, **kwargs: Dict[str, Any]):
        self.__type = type

    def __getattr__(self, name):
        def method(*args, **kwargs):
            if hasattr(self.client, name):
                return getattr(self.client, name)(*args, **kwargs)
            raise AttributeError(f"'{type(self.client).__name__}' has no attribute '{name}'")
        return method
    

    @property
    def type(self):
        return self.__type
