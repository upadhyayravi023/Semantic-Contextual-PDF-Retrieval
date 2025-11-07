# 📘 Semantic Contextual PDF Retrieval (SCPR)

This project is a full-stack, AI-powered web application that allows you to **"chat" with your PDF documents**.

You can upload any PDF, and the system will intelligently parse, index, and store its contents. Once indexed, you can ask questions in plain English, and the application will provide context-aware answers based only on the information contained within your document.

This application uses a **RAG (Retrieval-Augmented Generation)** pipeline, combining the power of Google's embedding models with its latest chat models to deliver accurate, cited answers.

---

## 🧩 How It Works

The application follows a two-stage **RAG (Retrieval-Augmented Generation)** process:

### 🧱 Indexing (Upload)
1. **Extract:** The user uploads a PDF. The backend extracts text using `pdfplumber` and a `pytesseract` OCR fallback for scanned images.  
2. **Chunk:** The full text is split into small, semantically-aware chunks.  
3. **Embed:** Each chunk is converted into a vector using Google’s `text-embedding-005` model.  
4. **Store:** These vectors are stored in a local **ChromaDB** vector database.  

### 🔍 Retrieval & Generation (Query)
1. **Retrieve:** The user’s question is embedded and searched against the vector database to find relevant text chunks.  
2. **Augment:** The relevant chunks (“context”) are combined with the user’s question to form a detailed prompt.  
3. **Generate:** The prompt is sent to Google’s `gemini-2.5-flash` model, which generates a context-aware answer.  

---

## ✨ Features

- 🧠 **Smart PDF Parsing** — Extracts text from both digital-native and scanned PDFs.  
- 📚 **Vector Indexing** — Uses `text-embedding-005` to understand semantic meaning.  
- 💾 **Persistent Vector Store** — Stores embeddings locally with **ChromaDB**.  
- 💬 **RAG Pipeline** — Generates accurate, context-based responses using Gemini.  
- ⚡ **Full-Stack App** — React frontend + Flask backend.  
- ✅ **Fully Tested** — Includes a pytest-based backend test suite.  

---

## 🧰 Tech Stack

### 🖥️ Frontend
- **React** — for SPA user interface.  
- **Axios** — for API requests.  

### ⚙️ Backend
- **Flask** — for API endpoints (`/upload`, `/query`).  
- **Colorama** — for color-coded console logs.  

### 🧠 AI & Data (LangChain Stack)
- **langchain-google-genai** — integration for Google AI models.  
- **ChatGoogleGenerativeAI (gemini-2.5-flash)** — for generating responses.  
- **GoogleGenerativeAIEmbeddings (text-embedding-005)** — for vectorization.  
- **langchain-chroma / chromadb** — local vector storage.  
- **langchain-text-splitters** — for chunking text.  

### 📄 PDF Processing
- **pdfplumber** — text extraction.  
- **PyMuPDF (fitz)** and **Pillow (PIL)** — image extraction.  
- **pytesseract** — OCR for scanned PDFs.  

### 🧪 Testing
- **pytest** — test framework.  
- **pytest-mock** — for mocking API calls.  

---

## ⚙️ Prerequisites

Before setup, ensure you have the following installed:

- 🐍 **Python (3.9+)** — [Download Python](https://www.python.org/downloads/)  
- 🧰 **Node.js (18+)** — [Download Node.js](https://nodejs.org/)  
- 🌀 **Git** — [Download Git](https://git-scm.com/)  
- 🔠 **Tesseract OCR Engine** — Required for OCR-based PDF text extraction  

```bash
# Windows
# Download and install from: https://github.com/tesseract-ocr/tesseract
```


## 🚀 Setup Instructions

## 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-project-repo.git
cd your-project-repo
```

## 2. Backend Setup (Flask Server)
```bash
# Navigate to the backend folder
cd backend

# Create and activate a Python virtual environment
# 🪟 On Windows
python -m venv venv
.\venv\Scripts\activate

# 🍎 / 🐧 On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create your environment file
# In backend/.env add:
GEMINI_API_KEY=your_google_api_key_here
```

## 3. Frontend Setup (React App)
 ```bash
# Navigate to the frontend folder
cd ../frontend

# Install NPM dependencies
npm install

# (Optional but recommended) Create a .env.local file
# In frontend/.env.local add:
REACT_APP_API_BASE_URL=http://127.0.0.1:5000
```
## Running the Application

# 🧠 Terminal 1: Run Backend (Flask)
```bash
cd backend
.\venv\Scripts\activate   # or source venv/bin/activate
python app.py
# Flask server → http://127.0.0.1:5000
```
# 💻 Terminal 2: Run Frontend (React)
```bash
cd frontend
npm start
# React app → http://localhost:3000
# Navigate to backend folder
cd backend
.\venv\Scripts\activate  # or source venv/bin/activate
```
# Install development dependencies
```bash
pip install -r requirements-dev.txt
```

# Run the test suite
```
pytest -v
```







