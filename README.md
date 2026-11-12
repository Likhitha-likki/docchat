# 📄 DocChat: RAG over your PDFs, with citations

Upload PDFs, ask questions, and get answers that cite the exact document and page. If the answer isn't in your documents, it says so instead of guessing.

### 👉 [**Live demo**](https://YOUR-APP-NAME.streamlit.app)

<!-- Add a screenshot or GIF after you deploy:  ![DocChat demo](docs/demo.png) -->

## Features

- **Cited answers:** every claim is tagged `[1]`, `[2]`, and the Sources panel shows the document, page and passage.
- **Hallucination control:** a relevance threshold stops the LLM from being called when nothing relevant is retrieved, and the prompt forbids outside knowledge.
- **Lightweight:** no PyTorch and no local LLM. Embeddings run on CPU via ONNX (`fastembed`), vectors live in FAISS, answers come from the Claude API.
- **Two interfaces:** a Streamlit chat UI and a FastAPI backend.
- **Tested offline:** the pipeline accepts injectable embedder / store / LLM, so tests need no API key or model download.

## How it works

```
PDF ──► loader (pypdf, per page)
          │
          ▼
       chunker (sentence-aware, ~800 chars, 150 overlap, keeps page number)
          │
          ▼
       embedder (fastembed · BAAI/bge-small-en-v1.5) ──► FAISS index
                                                              ▲
Question ──► embed query ──► top-k search ──► score filter ───┘
                                   │
                    nothing relevant? ──► "couldn't find that" (no LLM call)
                                   │
                                   ▼
              Claude (answers ONLY from context, cites [1] [2]...)
                                   │
                                   ▼
                   Answer + the sources it actually cited
```

## Tech stack

Python 3.12 · Claude API (Anthropic) · FAISS · fastembed (ONNX) · pypdf · Streamlit · FastAPI

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/docchat.git
cd docchat

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# edit .env and set ANTHROPIC_API_KEY=your-key

streamlit run app.py               # chat UI at http://localhost:8501
uvicorn api:app --reload           # REST API, docs at http://localhost:8000/docs
pytest                             # offline tests
```

Use Python 3.10-3.12. The first run downloads the embedding model (~130 MB).

## Configuration (environment variables or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Required |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Any Claude model; a smaller one is cheaper |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | Characters per chunk / overlap |
| `TOP_K` | 5 | Chunks sent to Claude |
| `MIN_SCORE` | 0.30 | Relevance cutoff; lower it if good questions return "not found" |
| `EMBED_CACHE_DIR` | system default | Where the embedding model is stored |
| `DOCCHAT_DATA_DIR` | `data` | Where the local index is saved |
| `DEMO_MODE` | off | `1` = private per-session index + question limit (for public hosting) |
| `DEMO_MAX_QUESTIONS` | 15 | Questions per session in demo mode |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | Upload a PDF (multipart field `file`) |
| POST | `/ask` | `{"question": "...", "top_k": 5}` returns the answer and cited sources |
| GET | `/documents` | Indexed documents and chunk counts |
| DELETE | `/reset` | Clear the index |

## Project structure

```
app.py               Streamlit UI
api.py               FastAPI backend
docchat/
  config.py          settings
  loader.py          PDF -> pages
  chunker.py         pages -> overlapping chunks
  embeddings.py      fastembed wrapper
  store.py           FAISS index + save/load
  llm.py             Claude prompt + streaming
  pipeline.py        ingest / retrieve / ask
tests/               chunker, loader, pipeline tests
```

## Deploy your own copy (Streamlit Community Cloud)

1. Fork or push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo, branch `main`, main file `app.py`.
3. In Advanced settings, choose Python 3.12 and add secrets:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   CLAUDE_MODEL = "claude-haiku-5-5"
   DEMO_MODE = "1"
   ```
4. Deploy. Set a spend limit on your Anthropic account, since anyone with the link can use your key.

## Design decisions

- **Page-level chunking:** chunks never span pages, so every citation is an exact page number.
- **Relevance gate:** below `MIN_SCORE` the LLM is never called, which reduces hallucination and cost.
- **Prompt-injection awareness:** document text is passed as untrusted context and Claude is told never to follow instructions inside it.
- **Verified citations:** only sources the answer actually cites are displayed.
- **Injectable components:** `DocChat(embedder=..., store=..., llm_stream=...)` keeps tests offline and makes it easy to evaluate retrieval.

## Limitations and roadmap

- [ ] OCR for scanned PDFs
- [ ] Multi-turn conversation with query rewriting
- [ ] Hybrid search (BM25 + vectors) and a reranker
- [ ] Retrieval evaluation (hit rate, MRR, faithfulness) across chunking strategies
- [ ] Better handling of tables and multi-column layouts
