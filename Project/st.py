# st.py

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title = "RAG Chat with OpenSearch",
    page_icon = "🔍",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
STATUS_ENDPOINT = f"{API_BASE_URL}/"

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .query-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #0d5d92;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html = True)

def check_api_status():
    """Checking if the FastAPI backend is running"""
    try:
        response = requests.get(STATUS_ENDPOINT, timeout = 5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def send_query(query):
    """Sending query to the FastAPI backend"""
    try:
        payload = {"query": query}
        response = requests.post(QUERY_ENDPOINT, json = payload, timeout = 30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return None, f"Connection error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def main():
    
    st.markdown('<h1 class="main-header">🔍 RAG Chat with OpenSearch</h1>', unsafe_allow_html = True)
    
    with st.sidebar:
        st.header("📊 System Status")
        
        # Checking API status
        status_placeholder = st.empty()
        is_online, status_data = check_api_status()
        
        if is_online:
            status_placeholder.success("✅ Backend API is online")
            if status_data:
                st.info(f"Status: {status_data.get('status', 'unknown')}")
                st.info(f"Message: {status_data.get('message', 'No message')}")
        else:
            status_placeholder.error("❌ Backend API is offline")
            st.warning("Please make sure the FastAPI server is running on port 8000")
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
           
        st.markdown("---")
        st.header("💡 Tips")
        st.markdown("""
        - Ask questions based on your uploaded documents
        - The system will search through your knowledge base
        - Responses are generated using AI with retrieved context
        - Complex questions may take longer to process
        """)
    
    st.header("💬 Chat Interface")
    
    # Initializing chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Displaying chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Adding user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Displaying user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Displaying assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔄 Thinking...")
            
            # Sending query to backend
            response, error = send_query(prompt)
            
            if error:
                full_response = f"❌ Error: {error}"
            elif response:
                full_response = response.get("answer", "No answer received")
            else:
                full_response = "No response received from server"
            
            # Simulating typing effect
            message_placeholder.markdown(full_response)
        
        # Adding assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    

    # Checking if an example query was clicked and simulate chat input
    if "chat_input" in st.session_state and st.session_state.chat_input:
        prompt = st.session_state.chat_input
        del st.session_state.chat_input
        
        # Adding user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Displaying user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Displaying assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔄 Thinking...")
            
            # Sending query to backend
            response, error = send_query(prompt)
            
            if error:
                full_response = f"❌ Error: {error}"
            elif response:
                full_response = response.get("answer", "No answer received")
            else:
                full_response = "No response received from server"
            
            # Simulating typing effect
            message_placeholder.markdown(full_response)
        
        # Adding assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()