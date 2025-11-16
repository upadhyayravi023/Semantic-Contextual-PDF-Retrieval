


import os
from flask import Flask
from flask.cli import load_dotenv
from flask_cors import CORS
from routes.upload_route import *
from routes.query_route import *
from core.embeddings import get_embedding_function
from app_init import create_app

load_dotenv()
app = create_app()
CORS(app) 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

UPLOAD_FOLDER = 'uploads'
DB_PATH = 'chroma_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DB_PATH'] = DB_PATH

vectorstore = None








if __name__ == '__main__':
    try:
        get_embedding_function()
        print("Embedding function initialized successfully.")
    except Exception as e:
        print(f"FATAL: {e}")
        print("Please ensure your GEMINI_API_KEY is correct in the .env file.")
        exit(1)


    app.run(debug=True, port=5000)













