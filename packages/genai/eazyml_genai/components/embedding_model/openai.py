from enum import Enum


class OpenAIEmbeddingModel(Enum):
    """
    Enumerates the supported OpenAI embedding models.

    Members:
        **TEXT_EMBEDDING_3_SMALL**: Represents the 'text-embedding-3-small' model.
        **TEXT_EMBEDDING_3_LARGE**: Represents the 'text-embedding-3-large' model.
        **TEXT_EMBEDDING_ADA_002**: Represents the 'text-embedding-ada-002' model.
    """
    TEXT_EMBEDDING_3_SMALL = 'text-embedding-3-small'
    TEXT_EMBEDDING_3_LARGE = 'text-embedding-3-large'
    TEXT_EMBEDDING_ADA_002 = 'text-embedding-ada-002'
