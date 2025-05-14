from enum import Enum
    

class GoogleEmbeddingModel(Enum):
    """
    Enumerates the supported Google embedding models.

    Members:
        **GEMINI_EMBEDDING_EXP_03_07**: Represents the 'gemini-embedding-exp-03-07' model.
        **TEXT_EMBEDDING_004**: Represents the 'text-embedding-004' model.
        **EMBEDDING_001**: Represents the 'embedding-001' model.
    """
    GEMINI_EMBEDDING_EXP_03_07 = 'gemini-embedding-exp-03-07'
    TEXT_EMBEDDING_004 = 'text-embedding-004'
    EMBEDDING_001 = 'embedding-001'
