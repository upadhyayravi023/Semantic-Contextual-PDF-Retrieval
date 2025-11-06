# import os
# import time
# import json
# from datetime import datetime
# from flask import Flask, request, jsonify
# from flask_cors import CORS 
# from dotenv import load_dotenv
# import pdfplumber
# import fitz  
# import io
# from PIL import Image
# import pytesseract


# def extract_text_from_pdf(filepath):
#     text = ""
    
#     try:
#         with pdfplumber.open(filepath) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
        
#         if not text.strip():
#             print("⚙️ Falling back to OCR (scanned PDF detected)...")
#             text = ""
#             doc = fitz.open(filepath)
#             for page_num in range(len(doc)):
#                 page = doc.load_page(page_num)
#                 pix = page.get_pixmap()
#                 img = Image.open(io.BytesIO(pix.tobytes("png")))
#                 ocr_text = pytesseract.image_to_string(img, lang="eng")
#                 text += ocr_text + "\n"

#         return text.strip()
    
#     except Exception as e:
#         print(f"❌ Error reading PDF: {e}")
#         return ""




# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from google import genai
# from google.genai.errors import APIError

# load_dotenv()
# app = Flask(__name__)
# CORS(app) 

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# UPLOAD_FOLDER = 'uploads'
# DB_PATH = 'chroma_db'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['DB_PATH'] = DB_PATH

# embedding_function = None
# vectorstore = None



# def get_embedding_function():
#     """Initializes the Gemini embedding function (transfer learning base)."""
#     global embedding_function
#     if embedding_function is None:
#         if not GEMINI_API_KEY:
#              raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")
#         try:
#            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY 
#            emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-005")


#         except Exception as e:
#             app.logger.error(f"Error initializing embedding function: {e}")
#             raise RuntimeError("Could not initialize embedding function.")
#     return embedding_function

# def initialize_vectorstore(file_path, file_name):
#     """Loads PDF, extracts text (OCR fallback), chunks it, and builds Chroma vectorstore."""
#     global vectorstore
#     start_time = time.time()

#     pdf_text = extract_text_from_pdf(file_path)
#     if not pdf_text.strip():
#         raise RuntimeError("No readable text found in PDF (even after OCR).")

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(pdf_text)
#     print(f"📄 PDF split into {len(chunks)} chunks.")
#     print(chunks)

#     if not chunks:
#         raise RuntimeError("No chunks created — PDF may be empty or unreadable.")

#     embedding_func = get_embedding_function()

#     vectorstore = Chroma.from_texts(
#         texts=chunks,
#         embedding=embedding_func,
#         persist_directory=app.config['DB_PATH']
#     )
#     vectorstore.persist()

#     end_time = time.time()

#     return {
#         "status": "success",
#         "file_name": file_name,
#         "total_chunks": len(chunks),
#         "indexing_duration": f"{end_time - start_time:.2f}s",
#         "timestamp": datetime.now().isoformat()
#     }



# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handles PDF upload and triggers the indexing process."""
#     if 'file' not in request.files:
#         return jsonify({"status": "error", "message": "No file part"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"status": "error", "message": "No selected file"}), 400

#     if file and file.filename.lower().endswith('.pdf'):
        
#         original_filename = file.filename
#         temp_filename = "document.pdf" 
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
#         file.save(file_path)

#         try:
#             metadata = initialize_vectorstore(file_path, original_filename)
#             return jsonify(metadata)
#         except RuntimeError as e:
           
#             return jsonify({"status": "error", "message": str(e)}), 500
#         except Exception as e:
#             app.logger.error(f"Indexing error: {e}")
#             return jsonify({"status": "error", "message": f"Failed to index PDF: {e}"}), 500
    
#     return jsonify({"status": "error", "message": "Invalid file type. Must be PDF."}), 400

# @app.route('/query', methods=['POST'])
# def query_document():
#     """Handles user query, performs RAG, and returns step-by-step JSON."""
#     global vectorstore
    
#     if vectorstore is None:
       
#         try:
#             embedding_func = get_embedding_function()
#             vectorstore = Chroma(persist_directory=app.config['DB_PATH'], embedding_function=embedding_func)
#             if vectorstore._collection.count() == 0:
#                  return jsonify({"status": "error", "message": "Vector store is empty. Please upload and index a PDF first."}), 400
#         except RuntimeError as e:
#             return jsonify({"status": "error", "message": str(e)}), 500
#         except Exception:
#             return jsonify({"status": "error", "message": "Vector store not initialized. Please upload a PDF."}), 400


#     data = request.get_json()
#     question = data.get('question', '')
#     if not question:
#         return jsonify({"status": "error", "message": "No question provided"}), 400

#     try:
#         start_overall = time.time()
       
#         start_retrieval = time.time()
        
#         retrieved_docs = vectorstore.similarity_search(question, k=5)
        
#         end_retrieval = time.time()
#         retrieval_duration = end_retrieval - start_retrieval

#         context_chunks = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
#         retrieved_documents_metadata = []
#         for doc in retrieved_docs:
#             metadata = doc.metadata
#             retrieved_documents_metadata.append({
#                 "source": metadata.get('source', 'Unknown'),
#                 "page": metadata.get('page', 'Unknown')
#             })
        
        
#         start_context_aug = time.time()
        
#         system_instruction = "You are an expert document analyst. Only answer questions using the provided context below. Do not use outside knowledge. If the answer is not in the context, state 'The required information is not available in the document.' Be concise."
      
#         prompt = f"{system_instruction}\n\nCONTEXT:\n{context_chunks}\n\nUSER QUESTION: {question}"
        
#         end_context_aug = time.time()
#         context_aug_duration = end_context_aug - start_context_aug

        
#         start_generation = time.time()
        
#         os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
#         llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=1.0
# )

        
#         response = llm.invoke(prompt)
#         final_answer = response.content

#         input_tokens = len(prompt.split())
#         output_tokens = len(final_answer.split())
         
        
#         end_generation = time.time()
#         generation_duration = end_generation - start_generation
#         end_overall = time.time()
#         overall_duration = end_overall - start_overall

#         return jsonify({
#             "status": "success",
#             "question": question,
#             "elapsed_time": {"overall": f"{overall_duration:.2f}s"},
#             "steps": [
#                 {
#                     "name": "Retrieval (Vector Search)",
#                     "status": "Completed",
#                     "duration": f"{retrieval_duration:.3f}s",
#                     "details": {
#                         "chunks_found": len(retrieved_docs),
#                         "retrieved_documents": retrieved_documents_metadata
#                     }
#                 },
#                 {
#                     "name": "Context Augmentation (Prompt Engineering)",
#                     "status": "Completed",
#                     "duration": f"{context_aug_duration:.3f}s",
#                     "details": {
#                         "model": "gemini-2.5-flash",
#                         "context_length_tokens_approx": input_tokens,
#                         "system_instruction_set": True
#                     }
#                 },
#                 {
#                     "name": "Generation (Gemini API Call)",
#                     "status": "Completed",
#                     "duration": f"{generation_duration:.3f}s",
#                     "details": {
#                         "input_tokens": input_tokens,
#                         "output_tokens_approx": output_tokens
#                     }
#                 }
#             ],
#             "final_answer": final_answer,
#             "source_context": context_chunks 
#         })

#     except APIError as e:
#         app.logger.error(f"Gemini API Error: {e}")
#         return jsonify({"status": "error", "message": f"Gemini API Error: {e}"}), 500
#     except Exception as e:
#         app.logger.error(f"An unexpected error occurred: {e}")
#         return jsonify({"status": "error", "message": f"An unexpected server error occurred: {e}"}), 500


# if __name__ == '__main__':
#     try:
#         get_embedding_function()
#         print("Embedding function initialized successfully.")
#     except Exception as e:
#         print(f"FATAL: {e}")
#         print("Please ensure your GEMINI_API_KEY is correct in the .env file.")
#         exit(1)


#     app.run(debug=True, port=5000)


import logging
from colorama import Fore, Style, init

# Initialize colorama for Windows/Unix
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)







import os
import time
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS 
from dotenv import load_dotenv
import pdfplumber
import fitz  
import io
from PIL import Image
import pytesseract


def extract_text_from_pdf(filepath):
    log.info(f"{Fore.CYAN}📄 Starting text extraction from PDF: {filepath}{Style.RESET_ALL}")
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            log.info(f"{Fore.YELLOW}→ PDF has {len(pdf.pages)} pages{Style.RESET_ALL}")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    log.info(f"{Fore.GREEN}✔ Extracted text from page {i+1}{Style.RESET_ALL}")
                    text += page_text + "\n"
                else:
                    log.warning(f"{Fore.MAGENTA}⚠ No text found on page {i+1}{Style.RESET_ALL}")
        
        if not text.strip():
            log.warning(f"{Fore.RED}⚙️ Falling back to OCR (scanned PDF detected)...{Style.RESET_ALL}")
            text = ""
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                text += ocr_text + "\n"
                log.info(f"{Fore.GREEN}🧠 OCR extracted text from page {page_num+1}{Style.RESET_ALL}")

        log.info(f"{Fore.CYAN}✅ Text extraction completed. Total characters: {len(text)}{Style.RESET_ALL}")
        return text.strip()

    except Exception as e:
        log.error(f"{Fore.RED}❌ Error reading PDF: {e}{Style.RESET_ALL}")
        return ""



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google import genai
from google.genai.errors import APIError

load_dotenv()
app = Flask(__name__)
CORS(app) 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

UPLOAD_FOLDER = 'uploads'
DB_PATH = 'chroma_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DB_PATH'] = DB_PATH

embedding_function = None
vectorstore = None



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


@app.route('/query', methods=['POST'])
def query_document():
    """Handles user query, performs RAG, and returns step-by-step JSON."""
    global vectorstore
    
    log.info(f"{Fore.CYAN}💬 Received query request{Style.RESET_ALL}")

    if vectorstore is None:
        log.warning(f"{Fore.YELLOW}⚠️ Vectorstore not loaded in memory. Attempting reload...{Style.RESET_ALL}")
        try:
            embedding_func = get_embedding_function()
            vectorstore = Chroma(persist_directory=app.config['DB_PATH'], embedding_function=embedding_func)
            if vectorstore._collection.count() == 0:
                log.error(f"{Fore.RED}❌ Vectorstore empty. Upload and index a PDF first.{Style.RESET_ALL}")
                return jsonify({"status": "error", "message": "Vector store is empty. Please upload and index a PDF first."}), 400
            else:
                log.info(f"{Fore.GREEN}✅ Vectorstore loaded successfully from {app.config['DB_PATH']}{Style.RESET_ALL}")
        except RuntimeError as e:
            log.error(f"{Fore.RED}❌ Embedding initialization failed: {e}{Style.RESET_ALL}")
            return jsonify({"status": "error", "message": str(e)}), 500
        except Exception as e:
            log.error(f"{Fore.RED}💥 Unknown error while reloading vectorstore: {e}{Style.RESET_ALL}")
            return jsonify({"status": "error", "message": "Vector store not initialized. Please upload a PDF."}), 400

    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        log.error(f"{Fore.RED}❌ No question provided in request{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": "No question provided"}), 400

    try:
        log.info(f"{Fore.CYAN}🧠 Processing query: '{question}'{Style.RESET_ALL}")
        start_overall = time.time()

        # Step 1: Retrieval
        start_retrieval = time.time()
        retrieved_docs = vectorstore.similarity_search(question, k=5)
        retrieval_duration = time.time() - start_retrieval
        log.info(f"{Fore.GREEN}🔍 Retrieved {len(retrieved_docs)} relevant chunks in {retrieval_duration:.3f}s{Style.RESET_ALL}")

        context_chunks = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        retrieved_documents_metadata = [
            {
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'Unknown')
            }
            for doc in retrieved_docs
        ]

        # Step 2: Context Augmentation
        start_context_aug = time.time()
        system_instruction = (
            "You are an expert document analyst. Only answer questions using the provided context below. "
            "Do not use outside knowledge. If the answer is not in the context, state "
            "'The required information is not available in the document.' Be concise."
        )
        prompt = f"{system_instruction}\n\nCONTEXT:\n{context_chunks}\n\nUSER QUESTION: {question}"
        context_aug_duration = time.time() - start_context_aug
        log.info(f"{Fore.BLUE}🧩 Context augmentation completed ({len(prompt.split())} tokens){Style.RESET_ALL}")

        # Step 3: Generation
        start_generation = time.time()
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)
        response = llm.invoke(prompt)
        final_answer = response.content
        generation_duration = time.time() - start_generation
        output_tokens = len(final_answer.split())

        log.info(f"{Fore.MAGENTA}🤖 Generated answer in {generation_duration:.3f}s ({output_tokens} tokens){Style.RESET_ALL}")

        overall_duration = time.time() - start_overall
        log.info(f"{Fore.GREEN}✅ Query completed successfully in {overall_duration:.2f}s{Style.RESET_ALL}")

        # Pretty summary banner
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"🧠 QUERY: {question}")
        print(f"🔍 CHUNKS RETRIEVED: {len(retrieved_docs)}")
        print(f"🗣️  OUTPUT TOKENS: {output_tokens}")
        print(f"⏱️  TOTAL TIME: {overall_duration:.2f}s")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        return jsonify({
            "status": "success",
            "question": question,
            "elapsed_time": {"overall": f"{overall_duration:.2f}s"},
            "steps": [
                {
                    "name": "Retrieval (Vector Search)",
                    "status": "Completed",
                    "duration": f"{retrieval_duration:.3f}s",
                    "details": {
                        "chunks_found": len(retrieved_docs),
                        "retrieved_documents": retrieved_documents_metadata
                    }
                },
                {
                    "name": "Context Augmentation (Prompt Engineering)",
                    "status": "Completed",
                    "duration": f"{context_aug_duration:.3f}s",
                    "details": {
                        "model": "gemini-2.5-flash",
                        "context_length_tokens_approx": len(prompt.split()),
                        "system_instruction_set": True
                    }
                },
                {
                    "name": "Generation (Gemini API Call)",
                    "status": "Completed",
                    "duration": f"{generation_duration:.3f}s",
                    "details": {
                        "output_tokens_approx": output_tokens
                    }
                }
            ],
            "final_answer": final_answer,
            "source_context": context_chunks
        })

    except APIError as e:
        log.error(f"{Fore.RED}🚨 Gemini API Error: {e}{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": f"Gemini API Error: {e}"}), 500

    except Exception as e:
        log.error(f"{Fore.RED}💥 Unexpected server error: {e}{Style.RESET_ALL}")
        return jsonify({"status": "error", "message": f"An unexpected server error occurred: {e}"}), 500


if __name__ == '__main__':
    try:
        get_embedding_function()
        print("Embedding function initialized successfully.")
    except Exception as e:
        print(f"FATAL: {e}")
        print("Please ensure your GEMINI_API_KEY is correct in the .env file.")
        exit(1)


    app.run(debug=True, port=5000)








