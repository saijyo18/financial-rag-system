# import os
# import streamlit as st
# from src.pdf_processor import load_and_process_pdfs
# from src.vector_store import initialize_vector_store, get_store_stats
# from src.rag_engine import generate_answer

# st.set_page_config(page_title="Financial RAG Assistant", layout="wide")
# st.title("📊 Quarterly Financial Reports RAG Engine")

# # Sidebar: Document Management
# st.sidebar.header("Document Management")
# uploaded_files = st.sidebar.file_uploader(
#     "Upload Quarterly Reports (PDF)", 
#     type=["pdf"], 
#     accept_multiple_files=True
# )

# if st.sidebar.button("Index Documents"):
#     if uploaded_files:
#         os.makedirs("data", exist_ok=True)
#         for uploaded_file in uploaded_files:
#             file_path = os.path.join("data", uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
        
#         with st.spinner("Extracting text and building vector embeddings..."):
#             chunks = load_and_process_pdfs("data")
#             store = initialize_vector_store(chunks)
#             st.sidebar.success(f"Successfully processed {len(uploaded_files)} files into {len(chunks)} chunks!")
#     else:
#         st.sidebar.warning("Please upload PDFs first.")

# # Display persistent database info
# try:
#     stats = get_store_stats()
#     st.sidebar.markdown("---")
#     st.sidebar.write("**Database Status:**")
#     st.sidebar.write(f"Total Chunks Stored: `{stats['total_chunks']}`")
# except Exception:
#     pass

# # Main UI: Query Section
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display conversation history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if "sources" in message:
#             with st.expander("View Cited Sources"):
#                 for idx, src in enumerate(message["sources"], 1):
#                     st.write(f"**{idx}. Document:** {src['file']} | **Page:** {src['page']}")

# # Handle user queries
# if query := st.chat_input("Ask a question about the quarterly reports..."):
#     if stats["total_chunks"] == 0:
#         st.error("Please upload and index documents before asking questions.")
#     else:
#         # Display user message
#         st.session_state.messages.append({"role": "user", "content": query})
#         with st.chat_message("user"):
#             st.markdown(query)

#         # Generate and display response
#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing financial documents..."):
#                 res = generate_answer(query)
#                 st.markdown(res["answer"])
                
#                 if res["sources"]:
#                     with st.expander("View Cited Sources"):
#                         for idx, src in enumerate(res["sources"], 1):
#                             st.write(f"**{idx}. Document:** {src['file']} | **Page:** {src['page']}")
                            
#         st.session_state.messages.append({
#             "role": "assistant", 
#             "content": res["answer"], 
#             "sources": res["sources"]
#         })

import os
import streamlit as st
from src.pdf_processor import load_and_process_pdfs
from src.vector_store import initialize_vector_store, get_store_stats
from src.rag_engine import generate_answer

st.set_page_config(page_title="Financial RAG Assistant", layout="wide")
st.title("📊 Quarterly Financial Reports RAG Engine")

# Sidebar: Document Management
st.sidebar.header("Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload Quarterly Reports (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True
)

if st.sidebar.button("Index Documents"):
    if uploaded_files:
        os.makedirs("data", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        with st.spinner("Extracting text and building vector embeddings..."):
            chunks = load_and_process_pdfs("data")
            store = initialize_vector_store(chunks)
            st.sidebar.success(f"Successfully processed {len(uploaded_files)} files into {len(chunks)} chunks!")
    else:
        st.sidebar.warning("Please upload PDFs first.")

# Display persistent database info
# FIX: give stats a safe default so the app never crashes if indexing
# hasn't happened yet in this session.
try:
    stats = get_store_stats()
except Exception:
    stats = {"total_chunks": 0}

st.sidebar.markdown("---")
st.sidebar.write("**Database Status:**")
st.sidebar.write(f"Total Chunks Stored: `{stats['total_chunks']}`")

# Main UI: Query Section
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # FIX: st.text() instead of st.markdown() — avoids "$...$" being
        # parsed as LaTeX math, which was garbling currency figures.
        st.text(message["content"])
        if "sources" in message:
            with st.expander("View Cited Sources"):
                for idx, src in enumerate(message["sources"], 1):
                    st.write(f"**{idx}. Document:** {src['file']} | **Page:** {src['page']}")

# Handle user queries
if query := st.chat_input("Ask a question about the quarterly reports..."):
    if stats["total_chunks"] == 0:
        st.error("Please upload and index documents before asking questions.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing financial documents..."):
                res = generate_answer(query)
                # FIX: st.text() instead of st.markdown() — same reason as above.
                st.text(res["answer"])
                
                if res["sources"]:
                    with st.expander("View Cited Sources"):
                        for idx, src in enumerate(res["sources"], 1):
                            st.write(f"**{idx}. Document:** {src['file']} | **Page:** {src['page']}")
                            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": res["answer"], 
            "sources": res["sources"]
        })