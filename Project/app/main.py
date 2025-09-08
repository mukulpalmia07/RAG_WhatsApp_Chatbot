# app/main.py

import os
import logging
from fastapi import FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn

from twilio.twiml.messaging_response import MessagingResponse
from app.twilio_client import TwilioClient

from app.retriever import LangChainOpenSearchRetriever
from app.rag import RAG
from dotenv import load_dotenv
load_dotenv()

'''
Creating a logger so we can see the error and module from which error came.
Make debugging and monitoring easier.
'''
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

''' These are our environment variables. '''
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    logger.warning("Missing GROQ_API_KEY in environment variables.")

if not OPENSEARCH_URL or not OPENSEARCH_INDEX or not EMBEDDING_MODEL or not GROQ_API_KEY:
    logger.error("Missing required env variables: OPENSEARCH_URL, OPENSEARCH_INDEX, EMBEDDING_MODEL, GROQ_API_KEY")
    raise ValueError("Required environment variables are missing")

''' These are our Twilio Client Credentials'''
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")

# Initializing Twilio client only if credentials are provided
if not TWILIO_SID and not TWILIO_AUTH and not TWILIO_WHATSAPP_FROM:
    logger.warning("Twilio credentials not provided - WhatsApp functionality disabled")


retriever = None
rag = None
twilio_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    ''' Initializes all the componenents on startup. '''
    
    global retriever, rag, twilio_client

    if TWILIO_SID and TWILIO_AUTH and TWILIO_WHATSAPP_FROM:
        try:
            twilio_client = TwilioClient(TWILIO_SID, TWILIO_AUTH, TWILIO_WHATSAPP_FROM)
            logger.info("Twilio client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Twilio: %s", e)
            twilio_client = None
    else:
        logger.warning("Twilio credentials missing - WhatsApp disabled")

    try:
        retriever = LangChainOpenSearchRetriever(
        index_name = OPENSEARCH_INDEX,
        opensearch_url = OPENSEARCH_URL,
        embedding_model = EMBEDDING_MODEL,
        space_type = "cosinesimil"
        )
        logger.info("OpenSearch Retriever initialized successfully.")

        if retriever.index_empty(OPENSEARCH_INDEX):
            logger.info("Index has 0 documents. Loading data.")
            retriever.load_from_folder("./data")
        else:
            logger.info("Index already has documents. Skipping load.")

        logger.info("Initializing RAG system...")
        rag = RAG(retriever = retriever)
        logger.info("RAG system Initialized successfully.")

        if rag.test_connection():
            logger.info("RAG system test passed - ready to serve requests")
        else:
            logger.warning("RAG system test failed - some functionality may not work")
            
        yield

    except Exception as e:
        logger.exception("Failed to initialize system components: %s", str(e))
        raise

app = FastAPI(title = "RAG using OpenSearch", lifespan = lifespan)

''' Defining a pydantic model, request and response. '''
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    status: str = "success"

class WhatsAppMessageRequest(BaseModel):
    to_number: str
    message: str


''' Simple GET endpoint used to verify the service is alive.'''
@app.get("/")
def get_status():
    try:
        if retriever is None or rag is None:
            return {"status": "unhealthy", "message": "System components not initialized"}
        
        return {"status": "healthy", "message": "RAG System is running", "services": {"rag": rag is not None, "twilio": twilio_client is not None}}
            
    except Exception as e:
        logger.exception("Status check failed: %s", str(e))
        return {"status": "error", "message": f"Status check failed: {str(e)}"}


@app.get("/test-send")
async def test_send():
    """Testing endpoint to send a WhatsApp message to your number"""
    
    if not twilio_client:
        raise HTTPException(status_code = 503, detail = "Twilio not configured")

    my_number = os.getenv("MY_WHATSAPP_NUMBER")
    if not my_number:
        raise HTTPException(
            status_code = 400, 
            detail = "MY_WHATSAPP_NUMBER environment variable not set"
        )
    
    try:
        result = twilio_client.send_whatsapp(
            to = my_number,
            body = "Test message from your RAG WhatsApp Bot!\n\nThis confirms your Twilio integration is working correctly!"
        )
        
        return {
            "status": "success", 
            "message": "Test message sent",
            "to": my_number,
            "message_id": result.get("sid") if result else "unknown"
        }
        
    except Exception as e:
        logger.error("Failed to send test message: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")
    

# Query endpoint
@app.post("/query", response_model = QueryResponse)
async def query(request: QueryRequest):
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code = 400, detail = "Query cannot be empty")
        
        if rag is None:
            raise HTTPException(status_code = 503, detail = "RAG system not initialized")
        
        logger.info("Processing query: %s", request.query[:100])
        
        # Using default values for top_k.
        result = rag.generate(
            query = request.query, 
            top_k = 5,  # Fixing default value
        )
        
        return QueryResponse(answer = result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed for input: %s", request.query)
        raise HTTPException(status_code = 500, detail = f"Query processing failed: {str(e)}")
    
# Twilio
@app.post("/webhook/twilio")
async def twilio_webhook(request: Request):
    try:
        form = await request.form()
        from_number = form.get("From")
        body = form.get("Body")
        logger.info("Incoming message from %s: %s", from_number, body)

        if not body or not body.strip():
            response = MessagingResponse()
            response.message("Please send a text message.")
            return Response(content = str(response), media_type = "application/xml")
    

        # Using the RAG system to generate response
        if rag is None:
            response = MessagingResponse()
            response.message("System not initialized yet. Please try again later.")
            return Response(content = str(response), media_type = "application/xml")

        try:
            # Using your RAG instance to generate the answer
            answer = rag.generate(query = body, top_k = 5)

        except Exception as e:
            logger.exception("RAG generation error")
            answer = "Sorry — I'm having trouble answering that right now. Please try again later."

        response = MessagingResponse()
        response.message(answer)
        return Response(content = str(response), media_type = "application/xml")

    except Exception as e:
        logger.exception("Webhook handler error: %s", e)
        twiml = MessagingResponse()
        twiml.message("An error occurred while processing your message. Please try again later.")
        return str(twiml)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host = os.getenv("HOST", "0.0.0.0"), 
        port = int(os.getenv("PORT", "8000")), 
        log_level = "info",
        reload = True
    )
