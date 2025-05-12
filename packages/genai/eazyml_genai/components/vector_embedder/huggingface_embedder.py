"""
Implements a VectorEmbedder for generating embeddings using HuggingFace Embedding API.
"""
import os
from enum import Enum
from turtle import st
from typing import Any, List, Union
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

from eazyml_genai.components.vector_embedder.vector_embedder import (
            VectorEmbedder,
            VectorEmbedderType
)

class HuggingfaceEmbedderModel(Enum):
    """
    Enumerates the supported Hugging Face Sentence Transformer models.

    Members:
        **ALL_MPNET_BASE_V2**: Represents the 'sentence-transformers/all-mpnet-base-v2' model.
        **ALL_MINILM_L6_V2**: Represents the 'sentence-transformers/all-MiniLM-L6-v2' model.
    """
    ALL_MPNET_BASE_V2 = 'sentence-transformers/all-mpnet-base-v2'
    ALL_MINILM_L6_V2 = 'sentence-transformers/all-MiniLM-L6-v2'
    CLIP_VIT_BASE_PATCH32 =  "openai/clip-vit-base-patch32"


class HuggingfaceEmbedderProcessor(Enum):
    CLIP_VIT_BASE_PATCH32 = 'openai/clip-vit-base-patch32'


class HuggingfaceEmbedder(VectorEmbedder):
    """
    A VectorEmbedder implementation for generating embeddings using Hugging Face Sentence Transformers.
    """
    def __init__(self, model: HuggingfaceEmbedderModel=None,
                 processor: HuggingfaceEmbedderProcessor=None,
                 **kwargs):
        """
        Initializes the HuggingfaceEmbedder with the specified model.

        Args:
            - **model** (`HuggingfaceEmbedderModel`): The Hugging Face Sentence Transformer model to use.
            - **kwargs: Additional keyword arguments.
        """
        super().__init__(type=VectorEmbedderType.HUGGINGFACE)
        if model:
            if model in [HuggingfaceEmbedderModel.CLIP_VIT_BASE_PATCH32]:
                model = CLIPModel.from_pretrained(model.value)
            elif model in [HuggingfaceEmbedderModel.ALL_MINILM_L6_V2,
                           HuggingfaceEmbedderModel.ALL_MPNET_BASE_V2]:
                model = SentenceTransformer(model.value)
            self.model = model
        else :
            self.model = None
        if processor:
            if processor in [HuggingfaceEmbedderProcessor.CLIP_VIT_BASE_PATCH32]:
                processor = CLIPProcessor.from_pretrained(processor.value, use_fast=True)
                self.processor = processor


    def generate_text_embedding(self, text: Union[str, List[str]],
                           batch_size: Union[int, None]=None,
                           **kwargs) -> Any:
        """
        Generates embeddings for the given text using the loaded Sentence Transformer model.

        Args:
            - **text** (`Union[str, List[str]]`): The text or list of texts to embed.
            - **batch_size** (`Union[int, None], optional`): The batch size for embedding generation. Defaults to None.
            - **kwargs**: Additional keyword arguments.

        Returns:
            The generated embeddings. The type of the embeddings depends on the Sentence Transformer model used.
        """
        if self.type == VectorEmbedderType.HUGGINGFACE:
            if batch_size is None :
                embeddings = self.model.encode(text)
            else :
                embeddings = self.model.encode(text, batch_size=batch_size)
            return embeddings


    def generate_image_embedding(self, image_path: Union[str, List[str]],
                           **kwargs) -> Any:
        if self.type == VectorEmbedderType.HUGGINGFACE:
            if isinstance(image_path, str):
                images = Image.open(image_path).convert("RGB")
            elif isinstance(image_path, List):
                images = [Image.open(i).convert("RGB") for i in image_path]
            inputs = self.processor(images=images, return_tensors="pt")
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # normalize
            return embeddings



    def embedding_size(self, model: HuggingfaceEmbedderModel) -> int:
        """
        Returns the embedding size for the specified Hugging Face model.

        Args:
            - **model** (`HuggingfaceEmbedderModel`): The Hugging Face Sentence Transformer model.

        Returns:
            (`int`): The embedding size for the model.
        """
        embedding_dict = {
            HuggingfaceEmbedderModel.ALL_MPNET_BASE_V2: 768,
            HuggingfaceEmbedderModel.ALL_MINILM_L6_V2: 384,
            HuggingfaceEmbedderModel.CLIP_VIT_BASE_PATCH32: 512
        }
        return embedding_dict[model]

