# Governed RAG Assistant (FREE Local)

This project is a governed, read-only Retrieval Augmented Generation (RAG) assistant.
It answers questions ONLY using approved documents (PDFs + Excel files).

## Architecture
User Question → Retriever (ChromaDB) → Top Chunks → Local LLM (Ollama + phi3) → Answer + Sources

## Folder Structure
governed_rag/
- app.py (Streamlit chatbot UI)
- ingest.py (builds vector DB from documents)
- requirements.txt
- data/pdf (put PDFs here)
- data/excel (put Excel files here)

## Setup 
### 1) Install Python packages
pip install -r requirements.txt

### 2) Install Ollama (Local LLM)
Download: https://ollama.com/download

### 3) Pull the model
ollama pull phi3

### 4) Build the vector database
python ingest.py

### 5) Run the chatbot
streamlit run app.py