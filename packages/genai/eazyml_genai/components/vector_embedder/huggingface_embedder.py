"""
Implements a VectorEmbedder for generating embeddings using HuggingFace Embedding API.
"""
import os
from abc import ABC, abstractmethod
from .embedding_model import (
    HuggingfaceEmbeddingModel,
    HuggingfaceEmbeddingProcessor
)
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

from .vector_embedder import (
            VectorEmbedder,
            VectorEmbedderType
)


class HuggingfaceEmbedder(VectorEmbedder):
    """
    A VectorEmbedder implementation for generating embeddings using Hugging Face Sentence Transformers.
    """
    def __init__(self, model: HuggingfaceEmbeddingModel=None,
                 processor: HuggingfaceEmbeddingProcessor=None,
                 **kwargs):
        """
        Initializes the HuggingfaceEmbedder with the specified model.

        Args:
            - **model** (`HuggingfaceEmbeddingModel`): The Hugging Face Sentence Transformer model to use.
            - **kwargs: Additional keyword arguments.
        """
        type=VectorEmbedderType.HUGGINGFACE
        model=model
        processor=processor
        if model:
            if model in [HuggingfaceEmbeddingModel.CLIP_VIT_BASE_PATCH32]:
                model = CLIPModel.from_pretrained(model.value)
            elif model in [HuggingfaceEmbeddingModel.ALL_MINILM_L6_V2,
                           HuggingfaceEmbeddingModel.ALL_MPNET_BASE_V2]:
                model = SentenceTransformer(model.value)
        if processor:
            if processor in [HuggingfaceEmbeddingProcessor.CLIP_VIT_BASE_PATCH32]:
                processor = CLIPProcessor.from_pretrained(processor.value, use_fast=True)
        super().__init__(
                type=type,
                model=model,
                processor=processor
                )

