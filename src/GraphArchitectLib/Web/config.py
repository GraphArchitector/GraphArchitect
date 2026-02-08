#"""
#Configuration module for GraphArchitect Web API.
#Centralizes all configuration values.
#"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"

# Database
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "grapharchitect.db"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT_START = int(os.getenv("PORT_START", "8000"))
PORT_END = int(os.getenv("PORT_END", "8010"))

# API
API_VERSION = "3.0.0"
API_PREFIX = "/api"

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# GraphArchitect Library
GRAPHARCHITECT_PATH = str(PROJECT_ROOT)

# Embedding
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
# Автоматический выбор: infinity если есть INFINITY_BASE_URL, иначе simple
_default_embedding = "infinity" if os.getenv("INFINITY_BASE_URL") else "simple"
EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", _default_embedding)
INFINITY_BASE_URL = os.getenv("INFINITY_BASE_URL", "http://localhost:7997")
INFINITY_API_KEY = os.getenv("INFINITY_API_KEY")
INFINITY_MODEL = os.getenv("INFINITY_MODEL", "BAAI/bge-small-en-v1.5")
INFINITY_TIMEOUT = int(os.getenv("INFINITY_TIMEOUT", "10"))

# k-NN Retriever
KNN_TYPE = os.getenv("KNN_TYPE", "naive")  # naive, faiss
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "FlatIP")  # FlatIP, FlatL2, HNSW
KNN_VECTOR_WEIGHT = float(os.getenv("KNN_VECTOR_WEIGHT", "0.7"))
KNN_TEXT_WEIGHT = float(os.getenv("KNN_TEXT_WEIGHT", "0.3"))

# Training
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
TEMPERATURE_CONSTANT = float(os.getenv("TEMPERATURE_CONSTANT", "1.0"))

# NLI
NLI_DATASET_PATH = DATA_DIR / "nli_examples.json"
NLI_K_EXAMPLES = int(os.getenv("NLI_K_EXAMPLES", "3"))
NLI_TYPE = os.getenv("NLI_TYPE", "llm")  # "knn", "qwen", "llm"
NLI_LLM_BACKEND = os.getenv("NLI_LLM_BACKEND", "openrouter")  # "openrouter", "vllm", "deepseek"
NLI_LLM_MODEL = os.getenv("NLI_LLM_MODEL", "openai/gpt-3.5-turbo")
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH")

# ReWOO Planning
USE_REWOO = os.getenv("USE_REWOO", "false").lower() == "true"
REWOO_MODEL = os.getenv("REWOO_MODEL", "google/gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
