# backend/routes/upload_route.py
import logging
import os
import time
from werkzeug.utils import secure_filename

from flask import Blueprint, request, jsonify
from colorama import Fore, Style, init as colorama_init

# initialize colorama (works on Windows)
colorama_init(autoreset=True)

# Basic logger for development
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Local imports (keep these as your project requires)
from core.embeddings import get_embedding_function
from core.vectorstore import initialize_vectorstore

upload_bp = Blueprint('upload_bp', __name__)

UPLOAD_FOLDER = 'uploads'
DB_PATH = 'chroma_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@upload_bp.route('/upload', methods=['POST'])
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

    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.pdf'):
        log.error(f"{Fore.RED}⚠️ Invalid file type uploaded. Only PDF allowed.{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": "Invalid file type. Must be PDF."}), 400

    temp_filename = filename  # keep original safe filename
    file_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        log.info(f"{Fore.CYAN}⬆️ File received: {filename}{Style.RESET_ALL}")
        file.save(file_path)
        log.info(f"{Fore.YELLOW}💾 File saved temporarily at: {file_path}{Style.RESET_ALL}")

        # Start indexing
        log.info(f"{Fore.CYAN}🚀 Starting indexing for {filename}...{Style.RESET_ALL}")
        start_time = time.time()
        metadata = initialize_vectorstore(file_path, filename)
        duration = time.time() - start_time

        # Ensure metadata has helpful fields
        if not isinstance(metadata, dict):
            metadata = {"status": "ok", "message": "Indexed", "total_chunks": 0}
        metadata.setdefault("indexing_duration", f"{duration:.2f}s")
        metadata.setdefault("total_chunks", metadata.get("total_chunks", 0))

        log.info(f"{Fore.GREEN}✅ Indexing completed successfully in {duration:.2f}s{Style.RESET_ALL}")
        log.info(f"{Fore.BLUE}📊 Indexed {metadata['total_chunks']} chunks from {filename}{Style.RESET_ALL}")
        log.info(f"{Fore.MAGENTA}📦 Vectorstore saved at: {DB_PATH}{Style.RESET_ALL}")

        # Optional console summary
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"📁 FILE INDEXED: {filename}")
        print(f"🧩 TOTAL CHUNKS: {metadata['total_chunks']}")
        print(f"⏱️  TIME TAKEN : {metadata['indexing_duration']}")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        return jsonify(metadata)

    except RuntimeError as e:
        log.exception(f"{Fore.RED}❌ Runtime error during indexing: {e}{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": str(e)}), 500

    except Exception as e:
        log.exception(f"{Fore.RED}💥 Unexpected indexing error: {e}{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": f"Failed to index PDF: {e}"}), 500
