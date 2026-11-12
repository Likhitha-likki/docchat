"""All settings in one place. Override any of them with environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DOCCHAT_DATA_DIR", "data"))
INDEX_DIR = DATA_DIR / "index"

# LLM
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# Embeddings (small ONNX model, no PyTorch needed)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR")  # set to a folder on E: to save C: space

# Chunking (characters, not tokens)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
# Chunks scoring below this cosine similarity are dropped. If nothing survives,
# we answer "not found" without calling the LLM. Tune this on your own documents.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))
