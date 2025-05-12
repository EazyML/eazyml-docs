"""
Implements a VectorEmbedder for generating embeddings using OpenAI's Embedding API.

Classes:
    OpenAIEmbedderModel: An enumeration of supported OpenAI embedding models.
    OpenAIEmbedder: A VectorEmbedder implementation for OpenAI models.

"""
import os
from enum import Enum
from typing import Any, List
from openai import OpenAI

from eazyml_genai.components.vector_embedder.vector_embedder import (
            VectorEmbedder,
            VectorEmbedderType
)

class OpenAIEmbedderModel(Enum):
    """
    Enumerates the supported OpenAI embedding models.

    Members:
        TEXT_EMBEDDING_3_SMALL: Represents the 'text-embedding-3-small' model.
        TEXT_EMBEDDING_3_LARGE: Represents the 'text-embedding-3-large' model.
        TEXT_EMBEDDING_ADA_002: Represents the 'text-embedding-ada-002' model.
    """
    TEXT_EMBEDDING_3_SMALL = 'text-embedding-3-small'
    TEXT_EMBEDDING_3_LARGE = 'text-embedding-3-large'
    TEXT_EMBEDDING_ADA_002 = 'text-embedding-ada-002'


class OpenAIEmbedder(VectorEmbedder):
    """
    A VectorEmbedder implementation for generating embeddings using OpenAI's Embedding API.
    """
    def __init__(self, model: OpenAIEmbedderModel, **kwargs):
        """
        Initializes the OpenAIEmbedder with the specified model and API key.

        Args:
            **model** (`OpenAIEmbedderModel`): The OpenAI embedding model to use.
            **kwargs** (`dict`): Additional keyword arguments, including 'api_key'. If 'api_key' is not provided,
                      it defaults to the 'OPENAI_API_KEY' environment variable.
        """
        super().__init__(type=VectorEmbedderType.OPENAI)
        api_key = kwargs.get('api_key', os.getenv('OPENAI_API_KEY'))
        client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        
    def generate_embedding(self, text: str, **kwargs) -> Any:
        """
        Generates an embedding for the given text using the specified OpenAI embedding model.

        Args:
            **text** (`str`): The text to embed.
            **kwargs** (`dict`): Additional keyword arguments to pass to the OpenAI embeddings API.

        Returns:
            The embedding response from the OpenAI API.
        """
        if self.type == VectorEmbedderType.OPENAI:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                **kwargs
            )
            return response
        
    def embedding_size(self, model: OpenAIEmbedderModel) -> int:
        """
        Returns the embedding size for the specified OpenAI embedding model.

        Args:
            **model** (`OpenAIEmbedderModel`): The OpenAI embedding model.

        Returns:
            The embedding size for the model.
        """
        embedding_dict = {
            OpenAIEmbedderModel.TEXT_EMBEDDING_3_SMALL : 1536,
            OpenAIEmbedderModel.TEXT_EMBEDDING_3_LARGE : 3072,
            OpenAIEmbedderModel.TEXT_EMBEDDING_ADA_002 : 1536,
        }
        return embedding_dict[model]

