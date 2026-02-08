

import sys
from pathlib import Path

# Добавляем пути
integration_path = Path(__file__).parent.parent
sys.path.insert(0, str(integration_path))

grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

#from GraphArchitectLib.Web.config import INFINITY_BASE_URL, INFINITY_API_KEY, INFINITY_TIMEOUT, INFINITY_MODEL, EMBEDDING_DIMENSION
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.embedding.infinity_embedding_service import InfinityEmbeddingService

# Simple
simple = SimpleEmbeddingService(
    dimension=1024, #324
    )
emb1_simple = simple.embed_text("Классифицировать текст")
emb2_simple = simple.embed_text("Категоризировать сообщение")
sim_simple = simple.compute_similarity(emb1_simple, emb2_simple)

print(f"Simple сходство: {sim_simple:.3f}")

# Infinity
infinity = InfinityEmbeddingService(
    base_url="http://localhost:7997",
    dimension=1024,
    model_name="BAAI/bge-m3"
)

emb1_inf = infinity.embed_text("Классифицировать текст")
emb2_inf = infinity.embed_text("Категоризировать сообщение")
sim_inf = infinity.compute_similarity(emb1_inf, emb2_inf)

print(f"Infinity сходство: {sim_inf:.3f}")
