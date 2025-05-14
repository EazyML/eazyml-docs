"""
Implements a VectorEmbedder for generating embeddings using Google's Generative AI API.

Classes:
    GoogleEmbeddingModel: An enumeration of supported Google embedding models.
    GoogleEmbedder: A VectorEmbedder implementation for Google models.

"""
import os
from enum import Enum
from typing import Any, List
from google import genai

from .embedding_model import GoogleEmbeddingModel

from .vector_embedder import (
            VectorEmbedder,
            VectorEmbedderType
)

    

class GoogleEmbedder(VectorEmbedder):
    """
    A VectorEmbedder implementation for generating embeddings using Google's Generative AI API.

    Attributes:
        **model** (`GoogleEmbeddingModel`): The Google embedding model to use.
    """
    
    def __init__(self, model: GoogleEmbeddingModel, **kwargs):
        """
        Initializes the GoogleEmbedder with the specified model and API key.

        Args:
            **model** (`GoogleEmbeddingModel`): The Google embedding model to use.
            **kwargs** (`dict`): Additional keyword arguments, including 'api_key'. If 'api_key' is not provided,
                      it defaults to the 'GEMINI_API_KEY' environment variable.
        """
        api_key = kwargs.get('api_key', os.getenv('GEMINI_API_KEY'))
        client = genai.Client(api_key=api_key)
        super().__init__(type=VectorEmbedderType.GOOGLE, model=model, client=client)
