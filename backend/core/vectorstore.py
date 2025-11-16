from cmath import log
from datetime import datetime
import time
import logging
log = logging.getLogger(__name__)


from colorama import Fore,Style
from flask import current_app as app
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google import genai
from google.genai.errors import APIError

from core.embeddings import get_embedding_function

from core.embeddings import get_embedding_function
from core.pdf_extractor import extract_text_from_pdf


def initialize_vectorstore(file_path, file_name):
    global vectorstore
    start_time = time.time()

    log.info(f"{Fore.CYAN}⚙️ Starting vectorstore initialization for: {file_name}{Style.RESET_ALL}")

    pdf_text = extract_text_from_pdf(file_path)
    if not pdf_text.strip():
        raise RuntimeError("No readable text found in PDF (even after OCR).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(pdf_text)
    log.info(f"{Fore.YELLOW}📚 PDF split into {len(chunks)} chunks{Style.RESET_ALL}")

    if not chunks:
        raise RuntimeError("No chunks created — PDF may be empty or unreadable.")

    embedding_func = get_embedding_function()
    log.info(f"{Fore.CYAN}💡 Creating embeddings and storing in Chroma DB...{Style.RESET_ALL}")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_func,
        persist_directory=app.config['DB_PATH']
    )
    vectorstore.persist()

    end_time = time.time()
    duration = end_time - start_time

    log.info(f"{Fore.GREEN}✅ Vectorstore successfully built in {duration:.2f}s{Style.RESET_ALL}")
    log.info(f"{Fore.BLUE}📦 Stored at: {app.config['DB_PATH']}{Style.RESET_ALL}")

    return {
        "status": "success",
        "file_name": file_name,
        "total_chunks": len(chunks),
        "indexing_duration": f"{duration:.2f}s",
        "timestamp": datetime.now().isoformat()
    }

