"""
Implements a VectorEmbedder for generating embeddings using OpenAI's Embedding API.

Classes:
    OpenAIEmbeddingModel: An enumeration of supported OpenAI embedding models.
    OpenAIEmbedder: A VectorEmbedder implementation for OpenAI models.

"""
import os
from enum import Enum
from typing import Any, List
from openai import OpenAI

from .embedding_model import (
    OpenAIEmbeddingModel,
)

from .vector_embedder import (
            VectorEmbedder,
            VectorEmbedderType
)


class OpenAIEmbedder(VectorEmbedder):
    """
    A VectorEmbedder implementation for generating embeddings using OpenAI's Embedding API.
    """
    def __init__(self, model: OpenAIEmbeddingModel, **kwargs):
        """
        Initializes the OpenAIEmbedder with the specified model and API key.

        Args:
            **model** (`OpenAIEmbeddingModel`): The OpenAI embedding model to use.
            **kwargs** (`dict`): Additional keyword arguments, including 'api_key'. If 'api_key' is not provided,
                      it defaults to the 'OPENAI_API_KEY' environment variable.
        """
        api_key = kwargs.get('api_key', os.getenv('OPENAI_API_KEY'))
        client = OpenAI(api_key=api_key)
        super().__init__(type=VectorEmbedderType.OPENAI, model=model, client=client)
        

