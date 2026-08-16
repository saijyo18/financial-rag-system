from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import os
import shutil
from src.pdf_processor import load_and_process_pdfs
from src.vector_store import initialize_vector_store, get_store_stats
from src.rag_engine import generate_answer

app = FastAPI(title="Financial RAG Service")

@app.post("/index")
async def index_documents(files: List[UploadFile] = File(...)):
    """Uploads PDFs, processes chunks, and updates vector store."""
    os.makedirs("data", exist_ok=True)
    saved_files = []
    
    for file in files:
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
        
    chunks = load_and_process_pdfs("data")
    initialize_vector_store(chunks)
    
    return {
        "status": "success",
        "processed_files": saved_files,
        "total_chunks_created": len(chunks)
    }

@app.get("/query")
async def query_rag(question: str, top_k: int = 4):
    """Executes RAG pipeline over persisted ChromaDB."""
    try:
        response = generate_answer(query=question, top_k=top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def store_statistics():
    """Returns collection statistics."""
    return get_store_stats()