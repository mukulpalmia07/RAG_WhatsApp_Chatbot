# app/retriever.py

import os
import logging
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredImageLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain.docstore.document import Document
from opensearchpy import OpenSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import logging

logging.getLogger("opensearch").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()


''' Creating a logger so we can see the error and module from which error came. '''
logger = logging.getLogger(__name__)

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not OPENSEARCH_URL or not OPENSEARCH_INDEX or not EMBEDDING_MODEL:
    logger.error("Missing required env variables: OPENSEARCH_URL, OPENSEARCH_INDEX, EMBEDDING_MODEL")
    raise ValueError("Required environment variables are missing")


''' Creating a class for managing vector storage and retrieval. '''
class LangChainOpenSearchRetriever:
    
    def __init__(
        self,
        index_name: str,
        opensearch_url: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        space_type: str = "cosinesimil"
    ):
        self.index_name = index_name
        self.opensearch_url = opensearch_url
        self.embedding_model = embedding_model
        self.embedding = HuggingFaceEmbeddings(model_name = self.embedding_model)

        # Connecting to OpenSearch
        self.os_client = OpenSearch(self.opensearch_url)
        
        # Creating index if not exists
        if not self.os_client.indices.exists(index = self.index_name):
            logger.info(f"Index {self.index_name} not found in OpenSearch. Creating it...")
            try:
                dim = len(self.embedding.embed_query("test"))
            except Exception:
                logger.warning("Failed to detect embedding dim, falling back to 384")
                dim = 384

            settings = {
                "settings": {
                    "number_of_shards": 1, 
                    "number_of_replicas": 0,
                    "index.knn": True
                },
                "mappings": {
                    "properties": {
                        "page_content": {"type": "text", "analyzer": "standard"},
                        "vector_field": {
                            "type": "knn_vector",
                            "dimension": dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": space_type,
                                "engine": "nmslib",
                            },
                        },
                        "metadata": {
                            "properties": {
                                "source": {"type": "keyword"},
                                "source_file": {"type": "keyword"},
                                "page": {"type": "integer"},
                                "page_label": {"type": "keyword"},
                                "total_pages": {"type": "integer"},
                                "lastpage": {"type": "integer"},
                                "chunk_id": {"type": "integer"},
                                "id": {"type": "keyword"},
                                "creationdate": {"type": "keyword"},  
                                "author": {"type": "keyword"},
                                "title": {"type": "text"}
                            }
                        }
                    }
                }
            }

            self.os_client.indices.create(index = self.index_name, body = settings)
            logger.info(f"Created index {self.index_name} with embedding dimension {dim}")

        # Initializing OpenSearchVectorSearch
        try:
            self.vectorstore = OpenSearchVectorSearch(
                index_name = self.index_name,  # Opensearch index where vector will be stored
                embedding_function = self.embedding,  # Converts text into embeddings before storing
                opensearch_url = self.opensearch_url,  # OpenSearch cluster endpoint
                space_type = space_type,  # Distance metric used for similarity
                text_field = "page_content",   
                vector_field = "vector_field"
            )
            logger.info(f"Connected to OpenSearch index {self.index_name} at {self.opensearch_url}")
        except Exception:
            logger.exception("Failed to connect to OpenSearch")
            raise

    def index_empty(self, index_name: str) -> bool:
        count = self.os_client.count(index = index_name)["count"]
        return count == 0


    def load_from_folder(self, folder_path: str = "./data"):
        """
        Load all files from a folder (txt, pdf, csv, images) and add them to OpenSearch.
        """
        docs = []
        ids = []
        metadatas = []

        if not os.path.exists(folder_path):
            logger.warning("Folder %s does not exist", folder_path)
            return


        for i, filename in enumerate(os.listdir(folder_path)):
            filepath = os.path.join(folder_path, filename)
            
            if not os.path.isfile(filepath):
                continue

            loader = None

            try:
                if filename.lower().endswith(".txt"):
                    loader = TextLoader(filepath, encoding="utf-8")

                elif filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(filepath)

                elif filename.lower().endswith(".csv"):
                    loader = CSVLoader(filepath)

                elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    loader = UnstructuredImageLoader(filepath)

                elif filename.lower().endswith(".docx"):
                    loader = Docx2txtLoader(filepath)

                elif filename.lower().endswith((".xls", ".xlsx")):
                    loader = UnstructuredExcelLoader(filepath)

                if loader:
                    logger.info(f"Loading document: {filename}")
                    loaded_docs = loader.load()
                    if not loaded_docs:
                        logger.warning("No documents loaded from file %s", filename)
                        continue
                    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
                    split_docs = splitter.split_documents(loaded_docs)

                logger.info(f"Document {filename} is being loaded with {len(split_docs)} chunks.")

                for j, d in enumerate(split_docs):
                    # Adding metadata for source file
                    d.metadata["source_file"] = filename
                    d.metadata["chunk_id"] = j + 1

                    doc_id = f"{filename}_chunk{j + 1}_{uuid.uuid4().hex[:8]}"
                    
                    docs.append(d.page_content)
                    ids.append(doc_id)
                    metadatas.append(d.metadata)

            except Exception as e:
                logger.exception("Failed to process file %s", filename)

        if docs:
            try:
                batch_size = 32
                for i in range(0, len(docs), batch_size):
                    batch_end = min(i + batch_size, len(docs))
                    try:
                        self.vectorstore.add_texts(
                            texts = docs[i : batch_end],
                            metadatas = metadatas[i : batch_end],
                            ids = ids[i : batch_end]
                        )

                    except Exception as e:
                        logger.exception("Failed to index batch %d-%d: %s", i, i + batch_end - 1, str(e))

                logger.info("Indexed %d docs into OpenSearch index %s", len(docs), self.index_name)
                
                # Ensuring new docs are immediately searchable
                self.os_client.indices.refresh(index = self.index_name)
                logger.info("Index %s refreshed.", self.index_name)

            except Exception:
                logger.exception("Failed to index documents")
        else:
            logger.warning("No docs found in folder %s", folder_path)
            

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        Perform hybrid search: BM25 + vector (semantic).
        alpha = weight for semantic score (0.5 = equal weight).
        """

        try:
            embedding = self.embedding.embed_query(query)

            body = {
                "size": k,
                "query": {
                    "hybrid": {
                        "queries": [
                            {
                                "match": {
                                    
                                    "text": {
                                        "query": query,
                                        "boost": 1.0 - alpha
                                    }
                                }
                            },
                            {
                                "knn": {
                                    "vector_field": {
                                        "vector": embedding,
                                        "k": k,
                                        "boost": alpha
                                    }
                                }
                            }
                        ]
                    }
                },
                "_source": ["text", "metadata"]
            }

            response = self.os_client.search(index = self.index_name, body = body)

        except Exception:
            logger.exception("Hybrid search failed")
            logger.info("Attempting fallback to vector-only search...")
            try:
                docs = self.vectorstore.similarity_search_with_score(query, k = k)
                results = []
                for doc, score in docs:
                    doc_id = doc.metadata.get("id", "unknown")
                    results.append((doc_id, doc.page_content, score, doc.metadata))
                return results
            except Exception:
                logger.exception("Fallback search also failed")
                raise   

        results = []
        for hit in response["hits"]["hits"]:
            doc_id = hit.get("_id", "unknown")
            text = hit["_source"].get("text")
            score = hit["_score"]
            metadata = hit["_source"].get("metadata", {})
            results.append((doc_id, text, score, metadata))

        return results