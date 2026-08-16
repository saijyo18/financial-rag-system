# import os

# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from src.vector_store import initialize_vector_store

# load_dotenv()

# # System prompt forcing grounding strictly on provided context
# PROMPT_TEMPLATE = """
# You are a precise financial analyst assistant. 
# Answer the user's question using ONLY the provided context snippets below. 

# Rules:
# 1. Always state numbers with their associated currency units and periods (e.g., "$10 million for Q1 FY26").
# 2. If the answer cannot be determined strictly from the provided context, respond EXACTLY with:
#    "I cannot answer this question based on the provided documents."
# 3. Do NOT guess or use outside knowledge.

# Context:
# {context}

# Question: {question}

# Answer:
# """

# def generate_answer(query: str, top_k: int = 4):
#     """Retrieves top_k relevant chunks and queries Gemini Flash 1.5."""
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise ValueError("GEMINI_API_KEY not found.")

#     # 1. Retrieve
#     vector_store = initialize_vector_store()
#     retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
#     retrieved_docs = retriever.invoke(query)

#     if not retrieved_docs:
#         return {
#             "answer": "I cannot answer this question based on the provided documents.",
#             "sources": []
#         }

#     # 2. Format Context
#     context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

#     # 3. Prompting
#     prompt = PromptTemplate(
#         template=PROMPT_TEMPLATE,
#         input_variables=["context", "question"]
#     )
#     formatted_prompt = prompt.format(context=context_str, question=query)

#     # 4. Generate with Gemini Flash
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-flash-latest",
#         temperature=0.0,
#         google_api_key=api_key
#     )
    
#     response = llm.invoke(formatted_prompt)

#     # Extract source metadata
#     sources = [
#         {
#             "file": doc.metadata.get("source", "Unknown"),
#             "page": doc.metadata.get("page", "Unknown")
#         }
#         for doc in retrieved_docs
#     ]

#     return {
#         "answer": response.content,
#         "sources": sources
#     }

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store import initialize_vector_store

load_dotenv()

PROMPT_TEMPLATE = """
You are a precise financial analyst assistant. 
Answer the user's question using ONLY the provided context snippets below. 

Rules:
1. Always state numbers with their associated currency units and periods (e.g., "$10 million for Q1 FY26").
2. If the answer cannot be determined strictly from the provided context, respond EXACTLY with:
   "I cannot answer this question based on the provided documents."
3. Do NOT guess or use outside knowledge.

Context:
{context}

Question: {question}

Answer:
"""


def extract_text(content):
    """
    Normalizes ChatGoogleGenerativeAI's response.content, which can be
    either a plain string or a list of content blocks (e.g. with Gemini
    'thought signature' metadata attached). Always returns clean text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()

    return str(content)


def generate_answer(query: str, top_k: int = 4):
    """Retrieves top_k relevant chunks and queries Gemini Flash."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")

    # 1. Retrieve
    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return {
            "answer": "I cannot answer this question based on the provided documents.",
            "sources": []
        }

    # 2. Format Context
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # 3. Prompting
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    formatted_prompt = prompt.format(context=context_str, question=query)

    # 4. Generate with Gemini Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.0,
        google_api_key=api_key
    )

    response = llm.invoke(formatted_prompt)
    answer_text = extract_text(response.content)

    # Extract source metadata
    sources = [
        {
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown")
        }
        for doc in retrieved_docs
    ]

    return {
        "answer": answer_text,
        "sources": sources
    }