"""
Defines an abstract base class for vector embedding generation, along with an enumeration
for supported vector embedder types.

Classes:
    VectorEmbedderType: An enumeration representing different vector embedder providers.
    VectorEmbedder: An abstract base class for vector embedding generation.

"""
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

class VectorEmbedderType(Enum):
    """
    Enumerates the supported types of vector embedding providers.

    Members:
        OPENAI: Represents the OpenAI embedding service.
        GOOGLE: Represents the Google embedding service.
        VERTEXAI: Represents the Vertex AI embedding service.
        HUGGINGFACE: Represents Hugging Face embedding models.
    """
    OPENAI = "openai"
    GOOGLE = "google"
    VERTEXAI = "vertexai"
    HUGGINGFACE = "huggingface"

class VectorEmbedder(ABC):
    """
    Abstract base class for vector embedding generation.

    Attributes:
        type (VectorEmbedderType): The type of vector embedder provider.

    Methods:
        __init__(type: VectorEmbedderType): Initializes the VectorEmbedder with the given type.
        generate_embedding(text: str) -> Any: Abstract method to generate an embedding for the given text.
    """
    def __init__(self, type: VectorEmbedderType):
        """
        Initializes the VectorEmbedder with the given type.

        Args:
            type (VectorEmbedderType): The type of vector embedder provider.
        """
        self.type = type
    
    