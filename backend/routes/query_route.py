import logging
import time
import os

from flask import jsonify, request, Blueprint
from config.settings import GEMINI_API_KEY
from colorama import Fore, Style

from core.embeddings import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.errors import APIError


log = logging.getLogger(__name__)

vectorstore = None   
UPLOAD_FOLDER = 'uploads'
DB_PATH = 'chroma_db'
query_bp = Blueprint("query_bp", __name__)

@query_bp.route('/query', methods=['POST'])
def query_document():
    """Handles user query, performs RAG, and returns step-by-step JSON."""
    global vectorstore
    
    log.info(f"{Fore.CYAN}💬 Received query request{Style.RESET_ALL}")

    if vectorstore is None:
        log.warning(f"{Fore.YELLOW}⚠️ Vectorstore not loaded in memory. Attempting reload...{Style.RESET_ALL}")
        try:
            embedding_func = get_embedding_function()
            vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_func)
            if vectorstore._collection.count() == 0:
                log.error(f"{Fore.RED}❌ Vectorstore empty. Upload and index a PDF first.{Style.RESET_ALL}")
                return jsonify({"status": "error", "message": "Vector store is empty. Please upload and index a PDF first."}), 400
            else:
                log.info(f"{Fore.GREEN}✅ Vectorstore loaded successfully from {DB_PATH}{Style.RESET_ALL}")
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
