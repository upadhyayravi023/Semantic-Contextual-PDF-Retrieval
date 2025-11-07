from asyncio import log
from datetime import time
import os
from tkinter.ttk import Style

from colorama import Fore
from flask import jsonify, request
from backend import app
from backend.core.embeddings import get_embedding_function
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google import genai
from google.genai.errors import APIError

from backend.core.embeddings import get_embedding_function
from backend.core.vectorstore import initialize_vectorstore



@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles PDF upload and triggers the indexing process."""
    log.info(f"{Fore.CYAN}📥 Received upload request{Style.RESET_ALL}")
    
    if 'file' not in request.files:
        log.error(f"{Fore.RED}❌ No file part in request{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        log.error(f"{Fore.RED}❌ No file selected for upload{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        original_filename = file.filename
        temp_filename = "document.pdf"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        log.info(f"{Fore.CYAN}⬆️ File received: {original_filename}{Style.RESET_ALL}")
        file.save(file_path)
        log.info(f"{Fore.YELLOW}💾 File saved temporarily at: {file_path}{Style.RESET_ALL}")

        try:
            log.info(f"{Fore.CYAN}🚀 Starting indexing for {original_filename}...{Style.RESET_ALL}")
            start_time = time.time()
            metadata = initialize_vectorstore(file_path, original_filename)
            duration = time.time() - start_time

            log.info(f"{Fore.GREEN}✅ Indexing completed successfully in {duration:.2f}s{Style.RESET_ALL}")
            log.info(f"{Fore.BLUE}📊 Indexed {metadata['total_chunks']} chunks from {original_filename}{Style.RESET_ALL}")
    
            log.info(f"{Fore.MAGENTA}📦 Vectorstore saved at: {app.config['DB_PATH']}{Style.RESET_ALL}")

            # Summary banner
            print(f"\n{Fore.GREEN}{'='*60}")
            print(f"📁 FILE INDEXED: {original_filename}")
            print(f"🧩 TOTAL CHUNKS: {metadata['total_chunks']}")    
            print(f"⏱️  TIME TAKEN : {metadata['indexing_duration']}")
            print(f"{'='*60}{Style.RESET_ALL}\n")

            return jsonify(metadata)
        
        except RuntimeError as e:
            log.error(f"{Fore.RED}❌ Runtime error during indexing: {e}{Style.RESET_ALL}")
            return jsonify({"status": "error", "message": str(e)}), 500
        
        except Exception as e:
            log.error(f"{Fore.RED}💥 Unexpected indexing error: {e}{Style.RESET_ALL}")
            return jsonify({"status": "error", "message": f"Failed to index PDF: {e}"}), 500
    
    else:
        log.error(f"{Fore.RED}⚠️ Invalid file type uploaded. Only PDF allowed.{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": "Invalid file type. Must be PDF."}), 400

