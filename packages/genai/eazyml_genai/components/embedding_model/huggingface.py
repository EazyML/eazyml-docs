from enum import Enum

    
class HuggingfaceEmbeddingModel(Enum):
    """
    Enumerates the supported Hugging Face Sentence Transformer models.

    Members:
        **ALL_MPNET_BASE_V2**: Represents the 'sentence-transformers/all-mpnet-base-v2' model.
        **ALL_MINILM_L6_V2**: Represents the 'sentence-transformers/all-MiniLM-L6-v2' model.
    """
    ALL_MPNET_BASE_V2 = 'sentence-transformers/all-mpnet-base-v2'
    ALL_MINILM_L6_V2 = 'sentence-transformers/all-MiniLM-L6-v2'
    CLIP_VIT_BASE_PATCH32 =  "openai/clip-vit-base-patch32"


class HuggingfaceEmbeddingProcessor(Enum):
    CLIP_VIT_BASE_PATCH32 = 'openai/clip-vit-base-patch32'
    
