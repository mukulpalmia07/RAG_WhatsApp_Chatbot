# app/rag.py

import os
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

''' Creating a logger so we can see the error and module from which error came. '''
logger = logging.getLogger(__name__)

''' Loading models from .env '''
GROQ_MODEL = os.getenv("GROQ_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_MODEL or not EMBEDDING_MODEL or not GROQ_API_KEY:
    logger.error("Missing required environment variables: GROQ_MODEL, EMBEDDING_MODEL, GROQ_API_KEY")
    raise ValueError("Required environment variables are missing")

''' Class to embed text using OpenAI'''
class Embedder:
    ''' 
    Creating a constructor that initializes an HuggingFace object.
    Storing it in self.embeddings.
    '''
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = EMBEDDING_MODEL
        )

    ''' 
    Creating a embedding method that takes a list of strings,
    Calls Huggingface to get embeddings and returns a list of embeddings (one vector per text)
    '''
    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            if not texts:
                return []
            else:
                return self.embeddings.embed_documents(texts)
            
        except Exception as e:
            logger.exception("HuggingFace embedding failed: %s", str(e))
            raise

''' 
Class to Retriever documents, format them as context, 
Pass then into LLMS with structured prompt and 
Return the final answer.
'''
class RAG:

    def __init__(self, retriever):
        """
        retriever -> instance of LangChainOpenSearchRetriever (from retriever.py)
        """
        try:
            self.retriever = retriever
            self.llm = ChatGroq(
                model = GROQ_MODEL,
                api_key = GROQ_API_KEY,
                temperature = 0.0, # Deterministic Answer, No Randomness.
                max_tokens = 1024 # Response Lenght
            )
            logger.info("Initialized ChatGroq with model: %s", GROQ_MODEL)
        except Exception as e:
            logger.exception("Failed to initialize ChatGroq: %s", str(e))
            raise

        # Creating Prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant that answers user questions using ONLY the provided context. "
                    "If the answer is not contained in the context, say you don't know and optionally provide guidance. "
                    "Be concise and accurate in your responses."
                ),
                (
                    "user",
                    "Context:\n{context}\n\n"
                    "User Question: {question}\n\n"
                    "Answer based only on the context provided above:"
                ),
            ]
        )


    def format_context(self, docs) -> str:

        """
        Formatting retrieved documents into context string for the LLM.
        Handles different document formats returned by hybrid_search.
        """
        formatted = []
        for i, d in enumerate(docs):
            try:
                if isinstance(d, tuple) and len(d) >= 3:  # (id, text, score, metadata)
                    doc_id, doc_text, score = d[0], d[1], d[2]
                    metadata = d[3] if len(d) > 3 else {}
                    
                    # Add source file info if available
                    source_info = ""
                    if metadata and "source_file" in metadata:
                        source_info = f" (Source: {metadata['source_file']})"
                    
                    formatted.append(f"[Document {i+1}]{source_info}\n{doc_text}")
                    
                elif isinstance(d, Document):
                    doc_id = d.metadata.get("id", f"doc_{i + 1}")
                    source_info = ""
                    if "source_file" in d.metadata:
                        source_info = f" (Source: {d.metadata['source_file']})"
                    formatted.append(f"[Document {i + 1}]{source_info}\n{d.page_content}")
                    
                else:
                    logger.warning(f"Skipping unexpected doc type: {type(d)}")
                    
            except Exception as e:
                logger.warning(f"Error formatting document {i}: {str(e)}")
                continue
                
        context = "\n\n---\n\n".join(formatted)
        logger.debug("Formatted context length: %d characters", len(context))
        return context


    def generate(self, query: str, top_k: int = 5) -> str:
      
        try:
            if not query or not query.strip():
                return "Please provide a valid question."
            
            logger.info("Processing query: %s", query[:100])  # Log first 100 chars
            
            # Retrieving docs from OpenSearch
            docs = self.retriever.hybrid_search(query, k = top_k)
            
            if not docs:
                return "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if the knowledge base contains relevant documents."

            # Building context string
            context_str = self.format_context(docs)
            
            if not context_str.strip():
                return "I found some documents but couldn't extract meaningful content from them."

            # Creating and running the chain
            try:
                chain = self.prompt | self.llm
                response = chain.invoke({"context": docs, "question": query})

                answer = response.content.strip()
                logger.info("Generated answer length: %d characters", len(answer))
                return answer
                
            except Exception as e:
                logger.exception("LLM chain execution failed: %s", str(e))
                return f"I encountered an error while generating the response: {str(e)}"

        except Exception as e:
            logger.exception("RAG generation failed: %s", str(e))
            return f"I encountered an error while processing your question: {str(e)}"


    def test_connection(self):
        """Test the RAG pipeline with a simple query."""
        try:
            test_query = "test"
            docs = self.retriever.hybrid_search(test_query, k = 1)
            logger.info("RAG test successful - retrieved %d documents", len(docs))
            return True
        except Exception as e:
            logger.exception("RAG test failed: %s", str(e))
            return False

