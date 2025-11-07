import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Folder Paths ---
# Define root directory for flexible paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
DB_PATH = os.path.join(ROOT_DIR, 'chroma_db')

# --- RAG Settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SIMILARITY_K = 5 # Number of chunks to retrieve

# --- LLM Settings ---
EMBEDDING_MODEL = "models/text-embedding-005"
GENERATIVE_MODEL = "gemini-2.5-flash"
TEMPERATURE = 1.0

# --- System Prompt ---
SYSTEM_INSTRUCTION = (
    "You are an expert document analyst. Only answer questions using the provided context below. "
    "Do not use outside knowledge. If the answer is not in the context, state "
    "'The required information is not available in the document.' Be concise."
)