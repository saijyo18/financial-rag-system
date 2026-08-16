# Financial RAG System — HSBC Holdings plc

A Retrieval-Augmented Generation (RAG) app that answers questions about quarterly/interim
financial reports, grounded strictly in the uploaded PDFs, with sources cited for every answer.

---

## 1. Documents Used
•	HSBC_Q1_2025
•	HSBC_Q2_2025
•	HSBC_Q3_2025
•	HSBC_Q4_2025
Source Links:
•	https://www.hsbc.com/investors/results-and-announcements/all-reporting/1q-2025-quick-read
•	https://www.hsbc.com/investors/results-and-announcements/all-reporting/3q-2025-quick-read

---

## 2. Setup Instructions

This project uses **Google Gemini** (free tier) for answer generation and **local
sentence-transformers** for embeddings — no OpenAI key required, and embeddings run
entirely offline with no rate limits.

### Prerequisites
- Python 3.10–3.12 recommended (3.14 works but is less battle-tested with this stack)
- A free Gemini API key from [aistudio.google.com](https://aistudio.google.com)

### Steps
```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd financial-rag-system

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
# Create a .env file in the project root with:
GEMINI_API_KEY=your_key_here

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload PDFs via the sidebar and click
**Index Documents**, or skip this step if `chroma_db/` already contains indexed data
from a previous run — the store persists to disk.

---

## 3. Architecture

| Component | Tool |
|---|---|
| PDF text extraction | `langchain_community.PyPDFLoader`, with file name, quarter label, and page number tracked per page, and prefixed directly into the chunk text (e.g. `[HSBC_Interim_Report_2026 - Page 4]`) |
| Chunking | `RecursiveCharacterTextSplitter` — 1200 char chunks, 200 char overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free, no rate limit) |
| Vector store | ChromaDB (persisted to `./chroma_db`) |
| Answer generation | Gemini (`gemini-flash-latest`) via `langchain_google_genai` |
| Interface | Streamlit |

### Why local embeddings instead of an API?
Initial development used Gemini's hosted embedding API (`gemini-embedding-001`), but
indexing ~1000 chunks repeatedly hit the free tier's 100-requests-per-minute quota
(`429 RESOURCE_EXHAUSTED`). Switched to local `sentence-transformers` embeddings, which
removed the rate limit entirely, cut indexing time from 20+ minutes to under a minute,
and required no API key at all for that step — at a small, acceptable cost to embedding
quality versus a larger hosted model.

---

## 4. Chunking Decision

- **Chunk size:** 1200 characters
- **Overlap:** 200 characters
- **Total chunks produced:** *(TODO — check your sidebar "Total Chunks Stored" value after a fresh index run)*
- **Reasoning:** 1200 characters was chosen at the upper end of the recommended range
  specifically to keep financial tables intact within a single chunk — financial press
  releases and interim reports are dense with tabular data (e.g. net fee income
  breakdowns by product/segment), and smaller chunk sizes were observed to split tables
  mid-row, losing the row's context. Each chunk is also prefixed with
  `[document_label - Page N]` before embedding, so the quarter/document identity is
  captured as part of the searchable text itself, not just in metadata — this avoids the
  common RAG failure mode where near-identical wording across different quarters gets
  confused during retrieval.

---

## 5. Prompt Used

```
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
```
Temperature: `0.0`

---

## 6. Honest Notes — What Didn't Work

- **Generic phrasing failed retrieval.** The question *"Tell me financial results of
  HSBC"* was refused (`"I cannot answer this question based on the provided documents"`),
  while the more document-language-aligned phrasing *"What was HSBC's net fee income for
  the half-year to 30 June 2026?"* retrieved correctly. This points to the chunk
  embeddings matching document vocabulary (e.g. "net fee income," "half-year") more
  closely than generic conversational phrasing ("financial results"). *(TODO — confirm
  this diagnosis by printing retrieved chunks for the failing question, and note the fix
  you applied — e.g. raising `top_k`, or rephrasing guidance in the UI.)*
- **`text-embedding-004` and `gemini-1.5-flash` are both fully deprecated** as of this
  project's build (mid-2026) — both returned 404s. Migrated to
  `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `gemini-flash-latest`
  (Google's auto-updating alias) for generation to avoid this recurring.
