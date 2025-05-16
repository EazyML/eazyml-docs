from abc import ABC
from typing import Dict, Any, Union
from enum import Enum

from sympy import N

from ..embedding_model.google import GoogleEmbeddingModel
from ..embedding_model.huggingface import HuggingfaceEmbeddingModel, HuggingfaceEmbeddingProcessor
from ..embedding_model.openai import OpenAIEmbeddingModel
from ..vector_embedder.google_embedder import GoogleEmbedder
from ..vector_embedder.huggingface_embedder import HuggingfaceEmbedder
from ..vector_embedder.openai_embedder import OpenAIEmbedder


class VectorDBType(Enum):
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


class VectorDB(ABC):
    
    text_embedding_model: Union[HuggingfaceEmbeddingModel,
                                GoogleEmbeddingModel,
                                OpenAIEmbeddingModel, None] = None
    image_embedding_model: Union[HuggingfaceEmbeddingModel, None] = None
    image_embedding_processor: Union[HuggingfaceEmbeddingProcessor, None] = None

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

    def init_embedding_models(self, **kwargs):
        # set text embedding model
        text_embedding_model = self.text_embedding_model
        if kwargs.get('text_embedding_model'):
            text_embedding_model = kwargs.get('text_embedding_model')
        elif hasattr(self, f'text_embedding_model'):
            if not self.text_embedding_model:
                text_embedding_model = HuggingfaceEmbeddingModel.ALL_MINILM_L6_V2
        self.text_embedding_model = text_embedding_model
        
        # set image embedding model
        image_embedding_model = self.image_embedding_model
        if kwargs.get('image_embedding_model'):
            image_embedding_model = kwargs.get('image_embedding_model')
        elif hasattr(self, f'image_embedding_model'):
            if not self.image_embedding_model:
                image_embedding_model = HuggingfaceEmbeddingModel.CLIP_VIT_BASE_PATCH32
        self.image_embedding_model = image_embedding_model
        
        
        # set image embedding processor
        image_embedding_processor = self.image_embedding_processor
        if kwargs.get('image_embedding_processor'):
            image_embedding_processor = kwargs.get('image_embedding_processor')
        elif hasattr(self, f'image_embedding_processor'):
            if not self.image_embedding_processor:
                image_embedding_processor = HuggingfaceEmbeddingProcessor.CLIP_VIT_BASE_PATCH32
        self.image_embedding_processor = image_embedding_processor
    
        
        # create client based on text embedding model
        if self.text_embedding_model:
            text_embed_client = None
            if isinstance(text_embedding_model, HuggingfaceEmbeddingModel):
                text_embed_client = HuggingfaceEmbedder(model=text_embedding_model)
            elif isinstance(text_embedding_model, OpenAIEmbeddingModel):
                text_embed_client = OpenAIEmbedder(model=text_embedding_model)
            elif isinstance(text_embedding_model, GoogleEmbeddingModel):
                text_embed_client = GoogleEmbedder(text_embedding_model)
            self.text_embed_client = text_embed_client
        
        # client based image embedding model, right now we don't have support for
        # image embedding from other provider
        if self.image_embedding_model:
            image_embed_client = HuggingfaceEmbedder(model=self.image_embedding_model,
                                                    processor=self.image_embedding_processor)
            self.image_embed_client = image_embed_client
        else :
            self.image_embed_client = None