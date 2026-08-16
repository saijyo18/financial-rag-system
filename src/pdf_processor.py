import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_process_pdfs(data_dir: str = "data") -> List[Document]:
    """
    Reads PDFs from the specified folder, extracts text with metadata,
    prefixes the content with the document quarter label, and chunks the text.
    """
    raw_documents = []
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    for pdf in pdf_files:
        file_path = os.path.join(data_dir, pdf)
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Add source context to every extracted page metadata and page content
        for page in pages:
            # Clean document label (e.g., Infosys_Q1_FY26)
            doc_label = os.path.splitext(pdf)[0]
            page.metadata["source"] = pdf
            page.metadata["quarter"] = doc_label
            page.metadata["page"] = page.metadata.get("page", 0) + 1  # 1-indexed page count
            
            # Prefix page content so vector search captures the quarter context
            page.page_content = f"[{doc_label} - Page {page.metadata['page']}]\n{page.page_content}"
            raw_documents.append(page)
            
    # Chunking strategy: 1200 chars preserves financial table integrity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    return chunks