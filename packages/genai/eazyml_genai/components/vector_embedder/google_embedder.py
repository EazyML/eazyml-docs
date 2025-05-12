"""
Implements a VectorEmbedder for generating embeddings using Google's Generative AI API.

Classes:
    GoogleEmbedderModel: An enumeration of supported Google embedding models.
    GoogleEmbedder: A VectorEmbedder implementation for Google models.

"""
import os
from enum import Enum
from typing import Any, List
from google import genai

from eazyml_genai.components.vector_embedder.vector_embedder import (
            VectorEmbedder,
            VectorEmbedderType
)

class GoogleEmbedderModel(Enum):
    """
    Enumerates the supported Google embedding models.

    Members:
        GEMINI_EMBEDDING_EXP_03_07: Represents the 'gemini-embedding-exp-03-07' model.
        TEXT_EMBEDDING_004: Represents the 'text-embedding-004' model.
        EMBEDDING_001: Represents the 'embedding-001' model.
    """
    GEMINI_EMBEDDING_EXP_03_07 = 'gemini-embedding-exp-03-07'
    TEXT_EMBEDDING_004 = 'text-embedding-004'
    EMBEDDING_001 = 'embedding-001'
    

class GoogleEmbedder(VectorEmbedder):
    """
    A VectorEmbedder implementation for generating embeddings using Google's Generative AI API.

    Attributes:
        **model** (`GoogleEmbedderModel`): The Google embedding model to use.
    """
    
    def __init__(self, model: GoogleEmbedderModel, **kwargs):
        """
        Initializes the GoogleEmbedder with the specified model and API key.

        Args:
            **model** (`GoogleEmbedderModel`): The Google embedding model to use.
            **kwargs** (`dict`): Additional keyword arguments, including 'api_key'. If 'api_key' is not provided,
                      it defaults to the 'GEMINI_API_KEY' environment variable.
        """
        super().__init__(type=VectorEmbedderType.GOOGLE)
        api_key = kwargs.get('api_key', os.getenv('GEMINI_API_KEY'))
        client = genai.Client(api_key=api_key)
        self.client = client
        self.model = model
        

    def generate_embedding(self, text: str, **kwargs) -> Any:
        """
        Generates an embedding for the given text using the specified Google embedding model.

        Args:
            **text** (`str`): The text to embed.
            **kwargs** (`dict`): Additional keyword arguments.

        Returns:
            The embedding response from the Google Generative AI API.
        """
        if self.type == VectorEmbedderType.GOOGLE:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
                )
            return response


    def embedding_size(self, model: GoogleEmbedderModel) -> int:
        """
        Returns the embedding size for the specified Google embedding model.

        Args:
            **model** (`GoogleEmbedderModel`): The Google embedding model.

        Returns:
            The embedding size for the model.
        """
        embedding_dict = {
            GoogleEmbedderModel.GEMINI_EMBEDDING_EXP_03_07 : 3072,
            GoogleEmbedderModel.TEXT_EMBEDDING_004 : 768,
            GoogleEmbedderModel.EMBEDDING_001 : 3072,
        }
        return embedding_dict[model]

