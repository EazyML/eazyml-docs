"""
Defines an abstract base class for vector embedding generation, along with an enumeration
for supported vector embedder types.

Classes:
    VectorEmbedderType: An enumeration representing different vector embedder providers.
    VectorEmbedder: An abstract base class for vector embedding generation.

"""
from abc import ABC, abstractmethod
from typing import Any, Union, List
from enum import Enum

from PIL import Image
import torch

from .embedding_model import(
        OpenAIEmbeddingModel,
        GoogleEmbeddingModel,
        HuggingfaceEmbeddingModel
)

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
    def __init__(self, type: VectorEmbedderType, model: Any=None, processor: Any=None, **kwargs):
        """
        Initializes the VectorEmbedder with the given type.

        Args:
            type (VectorEmbedderType): The type of vector embedder provider.
        """
        self.type = type
        self.model = model
        self.processor = processor
        self.client = kwargs.get('client')


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
            return embeddings.tolist()
        elif self.type == VectorEmbedderType.GOOGLE:
            if isinstance(self.model, GoogleEmbeddingModel):
                model_name = self.model.value
            elif isinstance(self.model, str) :
                model_name = self.model
            response = self.client.models.embed_content(
                model=model_name,
                contents=text
                )
            return response.embeddings[0].values
        elif self.type == VectorEmbedderType.OPENAI:
            if isinstance(self.model, OpenAIEmbeddingModel):
                model_name = self.model.value
            elif isinstance(self.model, str) :
                model_name = self.model
            response = self.client.embeddings.create(
                input=text,
                model=model_name
            )
            return response.data[0].embedding

    
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
            return embeddings.tolist()


    def embedding_size(self, model: Union[HuggingfaceEmbeddingModel,
                                          GoogleEmbeddingModel,
                                          OpenAIEmbeddingModel]) -> int:
        """
        Returns the embedding size for the specified Hugging Face model.

        Args:
            - **model** (`HuggingfaceEmbeddingModel`): The Hugging Face Sentence Transformer model.

        Returns:
            (`int`): The embedding size for the model.
        """
        embedding_size = -1
        if self.type == VectorEmbedderType.HUGGINGFACE:
            embedding_dict = {
                HuggingfaceEmbeddingModel.ALL_MPNET_BASE_V2: 768,
                HuggingfaceEmbeddingModel.ALL_MINILM_L6_V2: 384,
                HuggingfaceEmbeddingModel.CLIP_VIT_BASE_PATCH32: 512
            }
            embedding_size = embedding_dict[model]
        elif self.type == VectorEmbedderType.GOOGLE:
            embedding_dict = {
                GoogleEmbeddingModel.GEMINI_EMBEDDING_EXP_03_07 : 3072,
                GoogleEmbeddingModel.TEXT_EMBEDDING_004 : 768,
                GoogleEmbeddingModel.EMBEDDING_001 : 3072,
            }
            embedding_size = embedding_dict[model]
        elif self.type == VectorEmbedderType.OPENAI:
            embedding_dict = {
                OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL : 1536,
                OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE : 3072,
                OpenAIEmbeddingModel.TEXT_EMBEDDING_ADA_002 : 1536,
            }
            embedding_size = embedding_dict[model]
        return embedding_size

    