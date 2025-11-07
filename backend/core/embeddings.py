import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend import app
from backend.app import GEMINI_API_KEY


def get_embedding_function():
    """Initializes the Gemini embedding function (transfer learning base)."""
    global embedding_function
    if embedding_function is None:
        if not GEMINI_API_KEY:
             raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")
        try:
           os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY 
           emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-005")


        except Exception as e:
            app.logger.error(f"Error initializing embedding function: {e}")
            raise RuntimeError("Could not initialize embedding function.")
    return embedding_function