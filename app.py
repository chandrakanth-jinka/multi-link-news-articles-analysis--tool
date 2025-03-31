import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import validators
import huggingface_hub
import pickle
import time

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    st.error("⚠️ Hugging Face API key not found! Please add your API key to the .env file.")
    st.info("To get an API key, visit: https://huggingface.co/settings/tokens")
    st.stop()

# Try to validate the API key
try:
    huggingface_hub.whoami(token=api_key)
except Exception as e:
    st.error(f"Invalid Hugging Face API key. Please check your token. Error: {str(e)}")
    st.stop()

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQAWithSourcesChain

# Configure page
st.set_page_config(
    page_title="RockyBot: News Research Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "url_contents" not in st.session_state:
    st.session_state.url_contents = {}
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

def reformulate_query(query):
    """Reformulate the query to improve retrieval accuracy and semantic understanding."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["percentage", "stake", "share", "amount", "value"]):
        return f"""Find the exact numerical value or percentage in the context of: {query}
        Focus on specific numbers, percentages, or financial figures."""
    
    if any(word in query_lower for word in ["when", "date", "time", "period"]):
        return f"""Find the specific time, date, or period mentioned in relation to: {query}
        Look for temporal markers and chronological information."""
    
    if "why" in query_lower:
        return f"""Find the explanation, reason, or cause for: {query}
        Look for causal relationships and explanatory statements."""
    
    if "how" in query_lower:
        return f"""Find the detailed process, method, or steps regarding: {query}
        Look for sequential information and procedural details."""
    
    if any(word in query_lower for word in ["what", "who", "where"]):
        return f"""Find the specific factual information about: {query}
        Look for concrete details and explicit statements."""
    
    return f"""Find comprehensive information about: {query}
    Consider both explicit statements and implicit relationships."""

# Hugging Face model configuration
def get_llm():
    try:
        llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-large",
            huggingfacehub_api_token=api_key,
            task="text2text-generation",
            model_kwargs={
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True
            }
        )
        # Test the API connection with a simple prompt
        test_response = llm.invoke("Hello, this is a test message.")
        if test_response and isinstance(test_response, str):
            return llm
        else:
            st.error("Invalid response format from the model")
            return None
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.info("Please check your API key and make sure you have access to the model.")
        return None

# Get embeddings model
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

# Function to extract content from URLs
def extract_content_from_url(url):
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-upgrade',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if 'text/html' not in response.headers.get('Content-Type', ''):
            st.error(f"URL {url} does not contain HTML content")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove non-content elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'aside', 'noscript', 'meta', 'link']):
            element.decompose()
            
        # Try to find the main article content
        article = None
        selectors = [
            'article', '.article', '.post-content', '.entry-content', '.content',
            'main', '#main-content', '.main-content', '.story-content',
            '.article-content', '.news-content', '#content', '.body-content',
            '.article-body', '.post-body', '.entry-body', '.story-body',
            '.article-text', '.post-text', '.entry-text', '.story-text'
        ]
        
        for selector in selectors:
            article = soup.select_one(selector)
            if article:
                break
                
        if article:
            text = article.get_text()
        else:
            main = soup.find('main') or soup.find('div', class_=lambda x: x and any(term in x.lower() for term in ['content', 'article', 'story', 'news', 'body', 'text']))
            if main:
                text = main.get_text()
            else:
                text_blocks = [p.get_text() for p in soup.find_all('p')]
                if text_blocks:
                    text = ' '.join(text_blocks)
                else:
                    text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if not text:
            st.error(f"No text content found in URL {url}")
            return None
            
        # Add URL to the text for context
        text = f"Source URL: {url}\n\n{text}"
            
        return Document(page_content=text, metadata={"source": url})
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing URL {url}: {str(e)}")
        return None

# Function to process URLs and create vector store
def process_urls(urls):
    documents = []
    file_path = "faiss_store_openai.pkl"
    
    with st.spinner("Loading content from URLs..."):
        for url in urls:
            if not url.strip():
                continue
            
            try:
                doc = extract_content_from_url(url)
                if doc:
                    documents.append(doc)
                    st.session_state.url_contents[url] = True  # Track processed URLs
                    st.success(f"Successfully extracted content from {url}")
                else:
                    st.error(f"Failed to extract content from {url}")
            except Exception as e:
                st.error(f"Error processing {url}: {str(e)}")
    
    if not documents:
        st.error("No valid content found from the provided URLs.")
        return None
    
    try:
        with st.spinner("Processing content..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=['\n\n', '\n', '.', ','],
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                st.error("No text chunks created. Please check the content of your URLs.")
                return None
                
            st.info(f"Created {len(chunks)} text chunks for processing")
            
            embeddings = get_embeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            # Save the vector store
            with open(file_path, "wb") as f:
                pickle.dump(vector_store, f)
            
            return vector_store
    except Exception as e:
        st.error(f"Error during content processing: {str(e)}")
        return None

# Function to generate response
def generate_response(query, vector_store):
    llm = get_llm()
    if not llm:
        return "Error: Could not initialize the language model. Please check your API configuration."
    
    try:
        reformulated_query = reformulate_query(query)
        
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_kwargs={
                    "k": 4,
                    "score_threshold": 0.5
                }
            )
        )
        
        result = chain.invoke({"question": reformulated_query})
        
        if not result or "answer" not in result:
            return "Sorry, I couldn't generate a response. Please try rephrasing your question."
            
        answer = result["answer"]
        sources = result.get("sources", "")
        
        # Format the response with sources if available
        if sources:
            return f"{answer}\n\nSources:\n{sources}"
        return answer
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "An error occurred while generating the response. Please try again."

# Function to export chat history
def export_chat_history():
    chat_history = ""
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        chat_history += f"{role}: {message['content']}\n\n"
    
    buffer = BytesIO()
    buffer.write(chat_history.encode())
    buffer.seek(0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.txt"
    
    return buffer, filename

# UI Components
st.title("📈 RockyBot: News Research Tool")
st.markdown("Ask questions about content from multiple web links")

# URL input section
with st.sidebar:
    st.header("URL Configuration")
    
    if not st.session_state.processing_complete:
        url_count = st.number_input("How many links would you like to analyze?", 
                                    min_value=1, max_value=10, value=3, step=1)
        
        urls = []
        for i in range(url_count):
            url = st.text_input(f"URL {i+1}", key=f"url_{i}")
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                if not validators.url(url):
                    st.error(f"Invalid URL format for URL {i+1}")
                else:
                    urls.append(url)
        
        if st.button("Process URLs"):
            if urls:
                vector_store = process_urls(urls)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.processing_complete = True
                    st.rerun()
            else:
                st.error("Please enter at least one valid URL.")
    else:
        st.success("URLs processed successfully!")
        
        st.write("Processed URLs:")
        for i, url in enumerate(st.session_state.url_contents.keys()):
            st.write(f"{i+1}. {url}")
        
        if st.button("Process New URLs"):
            st.session_state.vector_store = None
            st.session_state.url_contents = {}
            st.session_state.processing_complete = False
            st.rerun()
        
        if st.session_state.messages:
            buffer, filename = export_chat_history()
            st.download_button(
                label="Export Chat History",
                data=buffer,
                file_name=filename,
                mime="text/plain"
            )

# Chat interface
if st.session_state.processing_complete:
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_query = st.chat_input("Ask a question about the content from the links...")
    
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(user_query, st.session_state.vector_store)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()  # Clear the input after submission
else:
    st.info("Please enter URLs in the sidebar and click 'Process URLs' to start.")

# Footer
st.markdown("---")
st.caption("Powered by Langchain, FAISS, and Hugging Face") 