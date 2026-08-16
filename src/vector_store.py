# # import os

# # from dotenv import load_dotenv
# # from langchain_chroma import Chroma
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # load_dotenv()

# # PERSIST_DIRECTORY = "./chroma_db"
# # COLLECTION_NAME = "financial_reports"

# # def get_embedding_model():
# #     """Initializes Google Gemini Free Embedding model."""
# #     api_key = os.getenv("GEMINI_API_KEY")
# #     if not api_key:
# #         raise ValueError("GEMINI_API_KEY not found in environment variables.")
# #     return GoogleGenerativeAIEmbeddings(
# #         model="models/gemini-embedding-001",
# #         google_api_key=api_key
# #     )

# # def initialize_vector_store(chunks=None):
# #     """
# #     Persists document chunks to ChromaDB with unique IDs to prevent duplicate entries.
# #     Returns the loaded Chroma instance.
# #     """
# #     embeddings = get_embedding_model()
    
# #     if chunks:
# #         # Create deterministic unique IDs based on file name, page, and chunk index
# #         ids = [
# #             f"{chunk.metadata.get('source', 'doc')}_p{chunk.metadata.get('page', 0)}_i{idx}"
# #             for idx, chunk in enumerate(chunks)
# #         ]
        
# #         vector_store = Chroma.from_documents(
# #             documents=chunks,
# #             embedding=embeddings,
# #             ids=ids,
# #             collection_name=COLLECTION_NAME,
# #             persist_directory=PERSIST_DIRECTORY
# #         )
# #     else:
# #         vector_store = Chroma(
# #             collection_name=COLLECTION_NAME,
# #             embedding_function=embeddings,
# #             persist_directory=PERSIST_DIRECTORY
# #         )
        
# #     return vector_store

# # def get_store_stats():
# #     """Returns database stats for restart verification."""
# #     store = initialize_vector_store()
# #     collection = store._collection
# #     return {
# #         "collection_name": COLLECTION_NAME,
# #         "total_chunks": collection.count(),
# #         "persistence_directory": PERSIST_DIRECTORY
# #     }

# import os
# import time

# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load_dotenv()

# PERSIST_DIRECTORY = "./chroma_db"
# COLLECTION_NAME = "financial_reports"
# BATCH_SIZE = 50      # stay well under the 100/min free-tier ceiling
# BATCH_DELAY = 65     # seconds between batches


# def get_embedding_model():
#     """Initializes Google Gemini Free Embedding model."""
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise ValueError("GEMINI_API_KEY not found in environment variables.")
#     return GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=api_key
#     )


# def initialize_vector_store(chunks=None):
#     """
#     Persists document chunks to ChromaDB with unique IDs to prevent duplicate entries.
#     Embeds in small batches with a delay to stay under the free-tier rate limit.
#     Returns the loaded Chroma instance.
#     """
#     embeddings = get_embedding_model()

#     if chunks:
#         ids = [
#             f"{chunk.metadata.get('source', 'doc')}_p{chunk.metadata.get('page', 0)}_i{idx}"
#             for idx, chunk in enumerate(chunks)
#         ]

#         vector_store = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=embeddings,
#             persist_directory=PERSIST_DIRECTORY
#         )

#         for i in range(0, len(chunks), BATCH_SIZE):
#             batch_chunks = chunks[i:i + BATCH_SIZE]
#             batch_ids = ids[i:i + BATCH_SIZE]

#             print(f"Embedding batch {i // BATCH_SIZE + 1} "
#                   f"({i + 1}-{i + len(batch_chunks)} of {len(chunks)} chunks)...")

#             vector_store.add_documents(documents=batch_chunks, ids=batch_ids)

#             if i + BATCH_SIZE < len(chunks):
#                 print(f"Pausing {BATCH_DELAY}s to stay under the free-tier rate limit...")
#                 time.sleep(BATCH_DELAY)

#     else:
#         vector_store = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=embeddings,
#             persist_directory=PERSIST_DIRECTORY
#         )

#     return vector_store


# def get_store_stats():
#     """Returns database stats for restart verification."""
#     store = initialize_vector_store()
#     collection = store._collection
#     return {
#         "collection_name": COLLECTION_NAME,
#         "total_chunks": collection.count(),
#         "persistence_directory": PERSIST_DIRECTORY
#     }

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "financial_reports"


def get_embedding_model():
    """Local, free, unlimited embedding model — no API key, no rate limits."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def initialize_vector_store(chunks=None):
    """
    Persists document chunks to ChromaDB with unique IDs to prevent duplicate entries.
    Returns the loaded Chroma instance.
    """
    embeddings = get_embedding_model()

    if chunks:
        ids = [
            f"{chunk.metadata.get('source', 'doc')}_p{chunk.metadata.get('page', 0)}_i{idx}"
            for idx, chunk in enumerate(chunks)
        ]

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            ids=ids,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
    else:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

    return vector_store


def get_store_stats():
    """Returns database stats for restart verification."""
    store = initialize_vector_store()
    collection = store._collection
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "persistence_directory": PERSIST_DIRECTORY,
    }