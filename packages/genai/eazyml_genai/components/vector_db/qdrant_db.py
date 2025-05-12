"""
A vector database wrapper for Qdrant, integrated into the EazyML GenAI framework.

This class initializes a connection to a Qdrant vector database using the provided
configuration parameters. It extends the `VectorDB` base class and sets the type
to `QDRANT`.
"""
from qdrant_client import QdrantClient, models
from ..vector_embedder.huggingface_embedder import (
            HuggingfaceEmbedder,
            HuggingfaceEmbedderModel,
            HuggingfaceEmbedderProcessor
)
from sklearn.feature_extraction.text import TfidfVectorizer
from .vector_db import VectorDB, VectorDBType
from ...license import (
        validate_license
)

class QdrantDB(VectorDB):
    """
    Attributes:
        client (QdrantClient): An instance of QdrantClient used for performing operations on the Qdrant database.

    Raises:
        ValueError: If required connection parameters are missing or incorrect.
    """
    def __init__(self, api_key: str=None, **kwargs):
        """
        Initializes a QdrantDB instance with the provided configuration.

        Args:
            - api_key (str): API key for secure access.
            - kwargs: Arbitrary keyword arguments used to configure the Qdrant client. These may include:
                - location (str, optional): Path to the local Qdrant instance.
                - url (str, optional): URL of the remote Qdrant instance.
                - port (int, optional): Port for HTTP communication.
                - grpc_port (int, optional): Port for gRPC communication.
                - https (bool, optional): Whether to use HTTPS.
                - api_key 
                - prefix (str, optional): API prefix path.
                - timeout (float or tuple, optional): Request timeout setting.
                - host (str, optional): Host address (legacy compatibility).
                - path (str, optional): Path to Qdrant data (legacy compatibility).
        """
        super().__init__(type=VectorDBType.QDRANT,
                         **kwargs)
        location = kwargs.get('location')
        url = kwargs.get('url')
        port = kwargs.get('port')
        grpc_port = kwargs.get('grpc_port')
        https = kwargs.get('https')
        api_key = kwargs.get('api_key')
        prefix = kwargs.get('prefix')
        timeout = kwargs.get('timeout')
        host = kwargs.get('host')
        path = kwargs.get('path')
        client = QdrantClient(location=location,
                                  url=url,
                                  port=port,
                                  grpc_port=grpc_port,
                                  https=https,
                                  api_key=api_key,
                                  prefix=prefix,
                                  timeout=timeout,
                                  path = path,
                                  host=host
                                  )
        self.client = client


    def delete_collection(self, collection_name):
        self.client.delete_collection(collection_name)
    
    
    def list_collection_names(self):
        collection_names = [collection.name for collection in self.client.get_collections().collections]
        return collection_names

        
    def create_collection(self, collection_name: str, **kwargs):
        if kwargs.get('override', False):
            self.delete_collection(collection_name=collection_name)
        if collection_name not in self.list_collection_names():
            text_embedding_model = kwargs.get('text_embedding_model', HuggingfaceEmbedderModel.ALL_MINILM_L6_V2)
            image_embedding_model = kwargs.get('image_embedding_model', HuggingfaceEmbedderModel.CLIP_VIT_BASE_PATCH32)
            image_embedding_processor = kwargs.get('image_embedding_processor', HuggingfaceEmbedderProcessor.CLIP_VIT_BASE_PATCH32)
            shard_number = kwargs.get('shard_number', 1)
            replication_factor = kwargs.get('replication_factor', 1)
            text_embed_client = HuggingfaceEmbedder(model=text_embedding_model)
            image_embed_client = HuggingfaceEmbedder(model=image_embedding_model,
                                                processor=image_embedding_processor)
            self.text_embed_client = text_embed_client
            self.image_embed_client = image_embed_client
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": models.VectorParams(
                                size=text_embed_client.embedding_size(model=text_embedding_model),  # Vector size is defined by used model
                                distance=models.Distance.COSINE,
                                datatype=models.Datatype.FLOAT16,
                                multivector_config=models.MultiVectorConfig(
                                        comparator=models.MultiVectorComparator.MAX_SIM
                                    ),
                            ),
                    "image": models.VectorParams(
                                size=image_embed_client.embedding_size(model=image_embedding_model),  # Vector size is defined by used model
                                distance=models.Distance.COSINE,
                                datatype=models.Datatype.FLOAT16,
                                multivector_config=models.MultiVectorConfig(
                                        comparator=models.MultiVectorComparator.MAX_SIM
                                    ),
                            )
                },
                sparse_vectors_config={"text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(datatype=models.Datatype.FLOAT16,
                                                    on_disk=True),
                    )},
                shard_number=shard_number,
                replication_factor=replication_factor
                )

    def index_documents(self,
                        collection_name,
                        documents,
                        **kwargs
                        ):
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform([f"{document['content']} {document['title']}"
                                                 if (document['content'] and document['title'])
                                                 else f"{document['content']}" for document in documents])
        self.vectorizer = vectorizer
        self.create_collection(collection_name,
                               **kwargs
                            )
        for idx, doc in enumerate(documents):
            vector_dict = {}
            if doc['type'] == 'text' and len(doc['path']) > 0:
                vector_dict['text'] = [
                    self.text_embed_client.generate_text_embedding(text=str(doc["title"])).tolist(),
                    self.text_embed_client.generate_text_embedding(text=str(doc["content"])).tolist()
                ]
                vector_dict['image'] = self.image_embed_client.generate_image_embedding(image_path=doc['path']).tolist()
            elif doc['type'] == 'text' and len(doc['path']) == 0:
                vector_dict['text'] = [
                    self.text_embed_client.generate_text_embedding(text=str(doc["title"])).tolist(),
                    self.text_embed_client.generate_text_embedding(text=str(doc["content"])).tolist()
                ]
            elif doc['type'] == 'image' and len(doc['path']) > 0:
                vector_dict['text'] = [
                    self.text_embed_client.generate_text_embedding(text=str(doc["content"])).tolist()
                ]
                vector_dict['image'] = self.image_embed_client.generate_image_embedding(image_path=doc['path']).tolist()
            elif doc['type'] == 'table'and len(doc['path']) > 0 :
                vector_dict['text'] = [
                    self.text_embed_client.generate_text_embedding(text=str(doc["content"])).tolist()
                ]
                vector_dict['image'] = self.image_embed_client.generate_image_embedding(image_path=doc['path']).tolist()
            else :
                print('doc', doc)
            if 'image' not in vector_dict:
                vector_dict['image'] = [[0]*self.image_embed_client.embedding_size(model=HuggingfaceEmbedderModel.CLIP_VIT_BASE_PATCH32)]
            
            # get sparse vector indices and values
            sparse_matrix = vectorizer.transform([f"{doc['content']} {doc['title']}" if (doc['content'] and doc['title']) else f"{doc['content']}"])
            row_indices, col_indices = sparse_matrix.nonzero()
            values = sparse_matrix.data
            vector_dict['text-sparse'] = models.SparseVector(
                            indices=col_indices,
                            values=values
                        )
            self.client.upsert(
                    collection_name=collection_name,
                    points=[models.PointStruct(
                                id=idx,
                                vector=vector_dict,
                                payload=doc
                            )
                    ]
                )
        print("indexed successfully")

    def retrieve_documents(self,
                           collection_name,
                           question,
                           top_k=10,
                           document_type='text'):
        total_hits = []
        # For dense vector search, we need to use the text embedding
        if document_type=='table':
            hits = self.client.query_points(
                        collection_name=collection_name,
                        query=[self.text_embed_client.generate_text_embedding(text=question, model=HuggingfaceEmbedderModel.ALL_MINILM_L6_V2).tolist()],
                        using='text',
                        query_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="type",
                                        match=models.MatchValue(
                                            value="table",
                                        ),
                                    )
                                ]
                        ),
                        limit=top_k,
                        with_vectors=False,
                        with_payload=True,
                    )
        else :
            hits = self.client.query_points(
                        collection_name=collection_name,
                        query=[self.text_embed_client.generate_text_embedding(text=question, model=HuggingfaceEmbedderModel.ALL_MINILM_L6_V2).tolist()],
                        using='text',
                        limit=top_k,
                        with_vectors=False,
                        with_payload=True,
                    )
        # add hit if its not there in total hit
        hits_ids = [hit.id for hit in total_hits]
        for hit in hits.points:
            if hit.id not in hits_ids:
                total_hits.append(hit)

        # For sparse vector search, we need to use the indices and values from the sparse matrix
        sparse_matrix = self.vectorizer.transform([question])
        _, col_indices = sparse_matrix.nonzero() # get row indices and column indices
        values = sparse_matrix.data
        if document_type=='table':
            hits = self.client.query_points(
                        collection_name=collection_name,
                        query=models.SparseVector(indices=col_indices, values=values),
                        using='text-sparse',
                        query_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="type",
                                        match=models.MatchValue(
                                            value="table",
                                        ),
                                    )
                                ]
                        ),
                        limit=top_k,
                        with_vectors=False,
                        with_payload=True,
                    )
        else :
            hits = self.client.query_points(
                        collection_name=collection_name,
                        query=models.SparseVector(indices=col_indices, values=values),
                        using='text-sparse',
                        limit=top_k,
                        with_vectors=False,
                        with_payload=True,
                    )


        # add hit if its not there in total hit
        hits_ids = [hit.id for hit in total_hits]
        for hit in hits.points:
            if hit.id not in hits_ids:
                total_hits.append(hit)
        return total_hits
