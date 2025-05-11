import os
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv  # Ensure dotenv is loaded

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Step 1: Load documents
logging.info("Loading documents from './data' folder...")
loader = DirectoryLoader('data', glob="./*.txt")
documents = loader.load()

# Step 2: Split documents into chunks
logging.info("Splitting documents into text chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings using HuggingFace model
logging.info("Loading HuggingFace embedding model: law-ai/InLegalBERT")
embedding_model = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

# Step 4: Create FAISS index and store document chunks
logging.info("Creating FAISS vector store...")
faiss_db = FAISS.from_documents(docs, embedding_model)

# Step 5: Save vector database locally
faiss_db.save_local("ipc_embed_db")
logging.info("Embedding database saved successfully at ./ipc_embed_db")

# Completion message
logging.info("Document ingestion and indexing complete.")
