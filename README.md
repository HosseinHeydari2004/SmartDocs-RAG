# SmartDocs-RAG

**A metadata-aware Retrieval-Augmented Generation (RAG) system for exploring medical and research documents.**

SmartDocs-RAG is a learning project focused on building a practical document-question-answering pipeline. It ingests documents, extracts and cleans their text, enriches chunks with metadata, stores embeddings in ChromaDB, and answers questions using retrieved passages with inline source references.

The project explores two important RAG concepts:

- **Semantic chunking** — splitting documents around semantic changes rather than relying only on a fixed character count.
- **Metadata filtering** — narrowing retrieval using document attributes such as author, publication year, document type, language, or filename.

> **Project status:** Educational / experimental. It is not a validated medical decision-support system and should not be used to make clinical decisions.

## Key features

- Upload digital documents such as PDF, DOCX, TXT, Markdown, and CSV.
- Process scanned PDFs and images using Tesseract OCR (optional system dependency).
- Clean extracted text before indexing.
- Extract document metadata with an LLM when configured.
- Split text using semantic chunking.
- Generate embeddings and store chunks in ChromaDB.
- Extract metadata constraints from a user's question and validate them before querying the vector store.
- Combine automatic metadata filters with manually selected filters.
- Generate answers grounded in retrieved passages and include inline citations such as `[1]`.
- Return an explicit “no sufficient evidence found” response when retrieval returns no matching passages.
- Expose a FastAPI backend and Streamlit interface.
- Track ingestion jobs and expose job status, progress, and cancellation endpoints.

## How it works

```text
                 ┌─────────────────────┐
                 │    Streamlit UI     │
                 └──────────┬──────────┘
                            │ HTTP
                 ┌──────────▼──────────┐
                 │      FastAPI        │
                 └──────────┬──────────┘
                            │
             ┌──────────────▼──────────────┐
             │       Ingestion pipeline    │
             │ Validate → Extract / OCR    │
             │ → Clean → Metadata          │
             │ → Semantic chunks           │
             │ → Embeddings → ChromaDB     │
             └─────────────────────────────┘

Question → Extract metadata filters → Validate filters
         → Filtered vector retrieval → LLM answer + sources
```

### 1. Document ingestion

The ingestion pipeline validates each file, extracts text (or uses OCR for scanned inputs), cleans the extracted content, attempts metadata extraction, creates semantic chunks, generates embeddings, and stores the resulting chunks in ChromaDB.

Each chunk can carry metadata such as:

- `filename` and `doc_id`
- `page_number` and document `kind`
- `title`, `author`, `year`
- `document_type` and `language`
- `ingested_at`

If metadata extraction fails, ingestion can continue with a warning rather than discarding the document.

### 2. Semantic chunking

The project uses LangChain's `SemanticChunker` with an embedding model to identify semantic breakpoints. This is intended to keep related ideas together more naturally than splitting text only at fixed character intervals.

Chunk quality depends on the source document, embedding model, and chunking configuration. Semantic chunking is not automatically better for every dataset, so its impact should be measured against a baseline.

### 3. Metadata-aware retrieval

When automatic filtering is enabled, the system attempts to identify constraints explicitly stated in the user's question. Supported fields include:

- Filename
- Author
- Document type
- Language
- Exact publication year, year range, or a list of years

The extracted filter is validated with Pydantic and translated into supported ChromaDB conditions. Manually selected filters take precedence for fields the user explicitly sets.

If filtering returns no matching passages, the system reports that it could not find sufficient evidence instead of silently removing the constraints and guessing.

### 4. Grounded answers and sources

Retrieved passages are passed to the language model as context. The model is instructed to answer using that context, cite passages inline, and say when the available documents do not contain enough evidence. The API also returns source metadata and retrieval information.

Citations help trace an answer back to retrieved material, but they do not guarantee that the answer is correct. Always verify important claims against the original documents.

## Tech stack

- **Language:** Python
- **API:** FastAPI, Uvicorn
- **UI:** Streamlit
- **RAG orchestration:** LangChain
- **Chunking:** LangChain SemanticChunker
- **Embeddings:** Sentence Transformers (`BAAI/bge-small-en-v1.5` by default)
- **Vector store:** ChromaDB
- **LLM provider:** OpenRouter
- **Metadata validation:** Pydantic
- **OCR:** Tesseract, PyTesseract, Pillow, and pdf2image

The default embedding model is English-focused. For Persian or multilingual corpora, configure a suitable multilingual embedding model and re-index the collection.

## Getting started

### Requirements

- Python 3.10+ (use a Python version supported by the installed dependencies)
- An OpenRouter API key for LLM-powered metadata extraction and answer generation
- Tesseract OCR and Poppler only if you need scanned-PDF/image OCR

### 1. Clone the repository

```bash
git clone https://github.com/HosseinHeydari2004/MedEvidence-RAG.git
cd MedEvidence-RAG
```

If your repository URL differs, replace the URL above with the correct one.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the LLM provider

Create a `.env` file in the project root and add your key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/auto
```

Never commit API keys or other secrets to version control.

### 5. Run the application

Start the Streamlit interface:

```bash
streamlit run ui/app.py
```

By default, the UI can start the local FastAPI backend automatically if it is not already running. The interface is available at `http://localhost:8501`, and the API documentation is available at `http://localhost:8000/docs`.

To run the API separately, disable UI auto-start and launch it in another terminal:

```bash
# Set SMARTDOCS_AUTOSTART_API=0 in your environment
uvicorn src.api.main:app --workers 1
```

Use one API worker because job state and session-supplied settings are held in process memory.

> **Note:** The repository's current UI/API configuration uses some `SmartDocs` naming (for example, `SMARTDOCS_AUTOSTART_API`). Keep those settings unchanged unless you update the code and configuration together.

## API overview

The API uses the `/api/v1` prefix for application endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Basic health check |
| `GET` | `/stats` | Vector-store and embedding statistics |
| `POST` | `/documents/upload` | Upload digital documents |
| `POST` | `/documents/upload-scanned` | Upload scanned documents for OCR |
| `GET` | `/jobs/{id}` | Check ingestion job status |
| `POST` | `/jobs/{id}/cancel` | Request cancellation of an ingestion job |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{doc_id}` | Delete an indexed document |
| `POST` | `/chat` | Ask a question over indexed documents |
| `GET` | `/settings/status` | Inspect runtime settings status |

Interactive API documentation is available at `/docs` when the backend is running.

## Testing

Run the test suite from the repository root:

```bash
pytest
```

The repository README reports 96 tests covering API behavior, ingestion, OCR-related behavior, metadata filters, RAG behavior, UI pages, and backend auto-start. Test availability and results may depend on the local environment and installed optional OCR tools.

## Current limitations

- The project has not been clinically validated and is not intended for diagnosis or treatment decisions.
- The default embedding model is English-focused; multilingual corpora need a suitable model and re-indexing.
- OCR extracts text but does not reliably preserve tables, formulas, charts, or complex layouts.
- OCR requires external Tesseract/Poppler installations and the relevant language packs.
- The API has no built-in authentication; do not expose it publicly without adding appropriate security controls.
- Job state and session-provided API settings are stored in memory and are lost when the process restarts.
- Retrieval quality, latency, cost, and answer faithfulness should be evaluated on a representative dataset before making performance claims.

## Learning goals and next steps

This project is part of my hands-on journey into RAG engineering. The current focus is on understanding how ingestion, semantic chunking, metadata extraction, metadata filtering, vector retrieval, and grounded generation work together.

Possible next experiments:

1. Compare semantic chunking with fixed-size and recursive character chunking.
2. Evaluate retrieval with Recall@k, Precision@k, and MRR on a curated question-answer set.
3. Evaluate generated answers for faithfulness and answer relevance.
4. Compare embedding models, chunk sizes, and retrieval settings.
5. Add hybrid retrieval and reranking, then measure whether they improve results.

## Disclaimer

MedEvidence-RAG is an educational software project. It may retrieve incomplete, outdated, or irrelevant information and may generate incorrect answers. It is not a substitute for professional medical judgment.

## Author

**Hossein Heydari**

- GitHub: [HosseinHeydari2004](https://github.com/HosseinHeydari2004)
