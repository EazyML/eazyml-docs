"""
A vector database wrapper for Pinecone, integrated into the EazyML GenAI framework.

This class provides an interface to interact with the Pinecone vector database
by extending the generic VectorDB class. It sets the vector DB type and initializes
a Pinecone client using the provided API key.
"""
import ast
import joblib
from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import (
        Pinecone,
        ServerlessSpec
)

from ...globals.settings import Settings
from ..vector_embedder.huggingface_embedder import (
            HuggingfaceEmbedder
)
from ..vector_embedder.google_embedder import (
            GoogleEmbedder
)
from ..vector_embedder.openai_embedder import (
            OpenAIEmbedder
)
from ..embedding_model import (
    OpenAIEmbeddingModel,
    GoogleEmbeddingModel,
    HuggingfaceEmbeddingModel,
    HuggingfaceEmbeddingProcessor
)
from sklearn.feature_extraction.text import TfidfVectorizer
from .vector_db import VectorDB, VectorDBType
from pinecone.enums import (
    Metric
)

from ...license import (
        validate_license
)

class PineconeDB(VectorDB):
    """
    Initializes the PineconeDB instance.

    Args:
        - **kwargs**: Dictionary of connection parameters like `url` and `api_key`.
        
    Example:
        .. code-block:: python

            # initialize pinecone vector database
            pinecone_db = PineconeDB(api_key=os.getenv("PINECONE_API_KEY"))
            
            # index document, mention collection name and documents
            # Give supported text embedding model from Hugginface, Google and OpenAI.
            indexed_documents = pinecone_db.index_documents(collection_name="USER DEFINED COLLECTION NAME",
                                    documents="JSON DOCUMENTS USING PDF LOADER",
                                    text_embedding_model=GoogleEmbeddingModel.TEXT_EMBEDDING_004,
                                    )
            
            # retrieve relevant document for given question.
            total_hits = pinecone_db.retrieve_documents("YOUR QUESTION", collection_name="YOUR COLLECTION NAME", top_k=5)
    """
    
    def __init__(self, **kwargs):
        super().__init__(type=VectorDBType.PINECONE,
                         **kwargs)
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("Missing 'api_key' in kwargs for PineconeDB initialization.")
        host = kwargs.get('host')
        if host and api_key:
            client = PineconeGRPC(
                api_key=api_key, 
                host=host,
            )
        elif api_key:
            client = Pinecone(api_key=api_key)
        self.client = client
        


    def list_collections(self):
        return self.client.list_indexes()
    
    def describe_collection(self, collection_name: str):
        if self.client.has_index(collection_name):
            return self.client.describe_index(name=collection_name)
        else:
            raise ValueError(f"No collection named {collection_name} exists")

        
    def delete_collection(self, collection_name: str):
        if self.client.has_index(collection_name):
            collection_description = self.describe_collection(collection_name)
            if collection_description.deletion_protection == "enabled":
                # If for some reason deletion protection is enabled, you will need to disable it first
                # before you can delete the index. But use caution as this operation is not reversible
                # and if somebody enabled deletion protection, they probably had a good reason.
                self.client.configure_index(name=collection_name, deletion_protection="disabled")
            self.client.delete_index(name=collection_name)
            
        
    def list_collection_names(self):
        collection_names = [collection["name"] for collection in self.client.list_indexes()]
        return collection_names
    
    # get dense sparse collection names for given collection name
    def get_ds_collection_names(self, collection_name):
        # create sparse collection
        sparse_collection_name = f"sparse-{collection_name}"
        # create dense collection
        dense_collection_name = f"dense-{collection_name}"
        return dense_collection_name, sparse_collection_name
        
    
    def create_dense_collection(self, dense_collection_name, **kwargs):
        if kwargs.get("overwrite", True):
            self.delete_collection(dense_collection_name)
        if dense_collection_name not in self.list_collection_names():
            self.delete_collection(dense_collection_name)
            metric = kwargs.get("metric", Metric.DOTPRODUCT)
            dimension = kwargs.get("dimension", self.text_embed_client.embedding_size(self.text_embedding_model))
            spec = kwargs.get("spec", ServerlessSpec(cloud="aws", region="us-east-1"))
            deletion_protection = kwargs.get("deletion_protection", "disabled")
            tags=kwargs.get("tags", {"environment": "development"})
            dense_index = self.client.create_index(
                name=dense_collection_name,
                vector_type="dense",
                dimension=dimension,
                metric=metric,
                spec=spec,
                deletion_protection=deletion_protection,
                tags=tags
            )
            return dense_index
        else :
            index_host = self.client.describe_index(name=self.dense_collection_name).host
            dense_index = self.client.Index(host=index_host)
            return dense_index
            
    def create_sparse_collection(self, sparse_collection_name, **kwargs):
        if kwargs.get("overwrite", True):
            self.delete_collection(sparse_collection_name)
        if sparse_collection_name not in self.list_collection_names():
            metric = kwargs.get("metric", Metric.DOTPRODUCT)
            spec = kwargs.get("spec", ServerlessSpec(cloud="aws", region="us-east-1"))
            deletion_protection = kwargs.get("deletion_protection", "disabled")
            tags=kwargs.get("tags", {"environment": "development"})
            sparse_index = self.client.create_index(
                name=sparse_collection_name,
                vector_type="sparse",
                metric=metric,
                spec=spec,
                deletion_protection=deletion_protection,
                tags=tags
            )
            return sparse_index
        else :
            index_host = self.client.describe_index(name=self.sparse_collection_name).host
            sparse_index = self.client.Index(host=index_host)
            return sparse_index
        

    def create_collection(self, collection_name, **kwargs):
        self.init_embedding_models(**kwargs)
        
        # get values for keyword argument using kwargs
        if 'spec' not in kwargs:
            spec = ServerlessSpec(
                cloud=kwargs.get('cloud', 'aws'),
                region=kwargs.get('region', "us-east-1")
            )
            # delete values of kwargs after getting the value otherwise,
            # it will throw multiple values for keyword argument
            if 'cloud' in kwargs:
                del kwargs['cloud']
            if 'region' in kwargs:
                del kwargs['region']
        else :
            spec=kwargs.get("spec")
            del kwargs['spec']
        tags = kwargs.get("tags", {"environment": "development"})
        
        
        # finally create index with specified parameters
        
        self.collection_name = collection_name
        dense_collection_name, sparse_collection_name = self.get_ds_collection_names(collection_name)
        
        # create sparse collection
        sparse_collection = self.create_sparse_collection(sparse_collection_name,
                                      metric=Metric.DOTPRODUCT,
                                      spec=spec,
                                      deletion_protection="disabled",
                                      tags=tags,
                                      **kwargs
                                      )
        # create dense collection
        dense_collection = self.create_dense_collection(dense_collection_name,
                                      metric=Metric.COSINE,
                                      dimension=self.text_embed_client.embedding_size(self.text_embedding_model),
                                      spec=spec,
                                      deletion_protection="disabled",
                                      tags=tags,
                                      **kwargs
                                      )
        return dense_collection, sparse_collection


    def index_sparse_documents(
        self,
        sparse_collection_name,
        records,
        namespace,
        **kwargs
    ):
        # initialize index
        index_host = self.client.describe_index(name=sparse_collection_name).host
        index = self.client.Index(host=index_host, grpc_config=GRPCClientConfig(secure=False))
        sparse_indexed = index.upsert(namespace=namespace, vectors=records)
        return sparse_indexed


    def index_dense_documents(
        self,
        dense_collection_name,
        records,
        namespace,
        **kwargs
    ):
        # initialize index
        index_host = self.client.describe_index(name=dense_collection_name).host
        index = self.client.Index(host=index_host, grpc_config=GRPCClientConfig(secure=False))
        dense_indexed = index.upsert(namespace=namespace, vectors=records)
        return dense_indexed

    def index_documents(self,
                        collection_name,
                        documents,
                        namespace="",
                        **kwargs
                        ):
        """Indexes a list of documents into separate dense and sparse vector collections.

        This method processes a list of dictionaries, where each dictionary represents a
        document and is expected to have at least a 'content' key and optionally a
        'title' key. It generates both dense embeddings using a text embedding client
        and sparse TF-IDF vectors for each document. These vectors, along with the
        document's metadata, are then stored in two separate collections (one for dense
        vectors and one for sparse vectors).

        Args:
            **collection_name** (`str`): 
                The base name for the dense and sparse vector
                collections that will be created or used. The actual collection names
                will be derived from this base name (e.g., 'my_collection_dense',
                'my_collection_sparse').
            
            **documents** (`list[dict]`): 
                A list of dictionaries, where each dictionary
                represents a document. Each document should have a 'content' key
                containing the text to be indexed. Optionally, a 'title' key can also
                be present and will be included in the text used for generating
                embeddings and sparse vectors. Any other key-value pairs in the
                document dictionary will be stored as metadata.
            
            **namespace** (`str`, `optional`): 
                An optional namespace to apply when indexing
                the documents into the collections. Defaults to "".
            
            **kwargs**: 
                Additional keyword arguments that will be passed to the
                `create_collection` method. This can include parameters like the
                dimension of the dense embeddings.

        """
        # initialize tfidf vectorizer for sparse vector embedding
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform([f"{document['content']} {document['title']}"
                                            if (document['content'] and document['title'])
                                            else f"{document['content']}" for document in documents])
        vectorizer_path = Settings.get_tfidfvectorizer_path(collection_name=collection_name)
        joblib.dump(vectorizer, vectorizer_path)
        
        # create seperate collection for dense and sparse collection
        dense_collection, sparse_collection = self.create_collection(
                                    collection_name,
                                    **kwargs
                                   )
        
        # create records using documents
        ids = list()
        dense_vectors = list()
        sparse_vectors = list()
        metadatas = list()
        for id, doc in enumerate(documents, start=1):
            ids.append(str(id))
            dense_vectors.append(self.text_embed_client.generate_text_embedding(text=f"{doc['content']} {doc['title']}"
                                                if (doc['content'] and doc['title'])
                                                else f"{doc['content']}"))
            # upsert sparse vectors
            sparse_matrix = vectorizer.transform([f"{doc['content']} {doc['title']}"
                                                if (doc['content'] and doc['title'])
                                                else f"{doc['content']}"])
            _, col_indices = sparse_matrix.nonzero()
            values = sparse_matrix.data
            sparse_vectors.append({"values": values, "indices": col_indices})
            metadatas.append({key: "" if value is None else str(value) for key, value in doc.items()})

        # create sparse records
        sparse_records = [{
                    "id": ids[i],
                    "sparse_values": sparse_vectors[i],
                    "metadata": metadatas[i]
                    } for i in range(len(ids))]
        
        # create dense records
        dense_records = [{
                    "id": ids[i],
                    "values": dense_vectors[i],
                    "metadata": metadatas[i]
                    } for i in range(len(ids))]
        
        
        dense_collection_name, sparse_collection_name = self.get_ds_collection_names(
                                                            collection_name=self.collection_name
                                                            )
        
        self.index_sparse_documents(
            sparse_collection_name=sparse_collection_name,
            records=sparse_records,
            namespace=namespace
        )
        
        self.index_dense_documents(
            dense_collection_name=dense_collection_name,
            records=dense_records,
            namespace=namespace
        )
        

    def retrieve_dense_documents(
        self,
        dense_collection_name,
        question,
        top_k=5,
        document_types=['text', 'table', 'image'],
        namespace="",
    ):
        # define index
        index_host = self.client.describe_index(name=dense_collection_name).host
        index = self.client.Index(host=index_host, grpc_config=GRPCClientConfig(secure=False))
        
        # retrieve results matching with title
        query_response = index.query(
            namespace=namespace,
            top_k=top_k,
            filter={
                "type": {"$in": document_types}
            },
            vector=self.text_embed_client.generate_text_embedding(question),
            include_values=False,
            include_metadata=True
        )
        return query_response
    
    def retrieve_sparse_documents(
        self,
        sparse_collection_name,
        vectorizer,
        question,
        top_k=5,
        document_types=['text', 'table', 'image'],
        namespace="",
    ):
        # define index
        index_host = self.client.describe_index(name=sparse_collection_name).host
        index = self.client.Index(host=index_host, grpc_config=GRPCClientConfig(secure=False))
        
        # retrieve results matching with title
        sparse_matrix = vectorizer.transform([f"{question}"])
        _, col_indices = sparse_matrix.nonzero()
        values = sparse_matrix.data
        query_response = index.query(
            namespace=namespace,
            top_k=top_k,
            filter={
                "type": {"$in": document_types}
            },
            sparse_vector={"values": values, "indices": col_indices},
            include_values=False,
            include_metadata=True
        )
        return query_response

    def retrieve_documents(self,
                           collection_name,
                           question,
                           top_k=5,
                           document_types=['text', 'table', 'image'],
                           namespace="",
                           **kwargs):
        """Retrieves documents relevant to a given question from both dense and sparse vector collections.
        This function performs a hybrid search, combining results from both dense and sparse
        vector retrieval methods to provide a more comprehensive set of relevant documents.
        It prevents duplicate documents from being returned.  It also transforms the metadata.

        Args:
            **question** (`str`): The query question used to retrieve relevant documents.
            
            **collection_name** (`str`, `optional`): The base name of the collections to query (both dense and sparse). If None, the default collection name (`self.collection_name`) is used. Defaults to None.
            
            **top_k** (`int`, `optional`): The number of top-ranking documents to retrieve from each collection (dense and sparse). Defaults to 5.
            
            **document_types** (`list[str]`, `optional`): A list of document types to filter the retrieval results.  Defaults to ['text', 'table', 'image'].
            
            **namespace** (`str`, `optional`): The namespace to use when querying the collections. Defaults to "".

        Returns:
            `list[dict]`: A list of retrieved documents. Each document is a dictionary containing the following keys:

                - 'id' (str): The unique identifier of the document.
                - 'score' (float): The relevance score of the document to the query.
                - 'metadata' (dict): A dictionary containing the document's metadata, including 'type', 'title', 'content', 'path' (converted from string representation), and 'meta' (converted from string representation). Empty strings are used if values are None.
        """
        self.init_embedding_models(**kwargs)
        vectorizer_path = Settings.get_tfidfvectorizer_path(collection_name=collection_name)
        vectorizer = joblib.load(vectorizer_path)
        
        if not collection_name:
            collection_name = self.collection_name
        dense_collection_name, sparse_collection_name = self.get_ds_collection_names(
                                                            collection_name=collection_name
                                                            )
        
        overall_documents = []
        documents_ids = []
        sparse_documents = self.retrieve_sparse_documents(
            sparse_collection_name=sparse_collection_name,
            vectorizer=vectorizer,
            question=question,
            top_k=top_k,
            document_types=document_types,
            namespace=namespace
        )
        
        for document in sparse_documents["matches"]:
            if document['id'] not in documents_ids:
                overall_documents.append(document)
                documents_ids.append(document['id'])
            
        dense_documents = self.retrieve_dense_documents(
            dense_collection_name=dense_collection_name,
            question=question,
            top_k=top_k,
            document_types=document_types,
            namespace=namespace
        )
        
        for document in dense_documents["matches"]:
            if document['id'] not in documents_ids:
                overall_documents.append(document)
                documents_ids.append(document['id'])
        
        for document in overall_documents:
            document['metadata'] = {
                    'type': document['metadata']['type'],
                    'title': document['metadata']['title'],
                    'content': document['metadata']['content'],
                    'path':ast.literal_eval(document['metadata']['path']),
                    'meta':ast.literal_eval(document['metadata']['meta']),
                    }
        return overall_documents
        
        
