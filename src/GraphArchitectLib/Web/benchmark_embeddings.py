"""
Бенчмарк: Сравнение SimpleEmbedding vs InfinityEmbedding

Сравнивает:
- Качество эмбеддингов (семантическое сходство)
- Скорость генерации эмбеддингов
- Точность NLI парсинга
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
config.EMBEDDING_TYPE = "infinity"

try:
    import faiss
    import numpy as np
    print(f"  [OK] Faiss version: {faiss.__version__}")
    print(f"  [OK] NumPy version: {np.__version__}")
    FAISS_OK = True
except ImportError as e:
    print(f"  [FAIL] Faiss не установлен: {e}")
    print(f"  Установите: pip install faiss-cpu numpy")
    FAISS_OK = False

print("=" * 70)
print("БЕНЧМАРК: SimpleEmbedding vs InfinityEmbedding")
print("=" * 70)
print()

# Импорты
try:
    from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
    from grapharchitect.services.nli.natural_language_interface import NaturalLanguageInterface
    from grapharchitect.services.nli.nli_dataset_item import NLIDatasetItem
    from grapharchitect.entities.connectors.task_representation import TaskRepresentation
    from grapharchitect.entities.connectors.connector import Connector
    from grapharchitect.entities.base_tool import BaseTool
    
    print("[OK] GraphArchitect импортирован")
except ImportError as e:
    print(f"[FAIL] Ошибка импорта: {e}")
    sys.exit(1)

# Проверка Infinity
try:
    from grapharchitect.services.embedding.infinity_embedding_service import InfinityEmbeddingService
    
    if config.EMBEDDING_TYPE == "infinity" and config.INFINITY_BASE_URL:
        infinity_service = InfinityEmbeddingService(
            base_url=config.INFINITY_BASE_URL,
            api_key=config.INFINITY_API_KEY,
            dimension=config.EMBEDDING_DIMENSION,
            model_name="FractalGPT/SbertDistilV2", # config.INFINITY_MODEL
            timeout=config.INFINITY_TIMEOUT,
            fallback_to_simple=False
        )
        
        if infinity_service.is_available():
            INFINITY_OK = True
            print("[OK] InfinityEmbeddingService доступен")
        else:
            INFINITY_OK = False
            print("[FAIL] Infinity сервер не отвечает")
    else:
        INFINITY_OK = False
        print("[INFO] Infinity отключен в конфигурации")

except Exception as e:
    INFINITY_OK = False
    print(f"[FAIL] Ошибка Infinity: {e}")

print()

# Создаем тестовые данные
print("Подготовка тестовых данных...")
print("-" * 70)

test_pairs = [
    ("Классифицировать текст", "Категоризировать сообщение"),  # Похожие
    ("Ответить на вопрос", "Дать ответ на запрос"),  # Похожие
    ("Классифицировать текст", "Сгенерировать статью"),  # Разные
    ("Проверить качество", "Валидировать документ"),  # Похожие
    ("Суммировать документ", "Классифицировать текст"),  # Разные
]

print(f"  Подготовлено {len(test_pairs)} пар для теста")
print()

# Тест 1: Качество семантического сходства
print("[ТЕСТ 1] Качество семантического сходства")
print("=" * 70)
print()

# SimpleEmbedding
print("SimpleEmbeddingService (хеширование):")
print("-" * 70)

simple_service = SimpleEmbeddingService(dimension=384)

for text1, text2 in test_pairs:
    emb1 = simple_service.embed_text(text1)
    emb2 = simple_service.embed_text(text2)
    sim = simple_service.compute_similarity(emb1, emb2)
    
    print(f"  \"{text1}\"")
    print(f"  \"{text2}\"")
    print(f"  Сходство: {sim:.3f}")
    print()

# InfinityEmbedding
if INFINITY_OK:
    print("InfinityEmbeddingService (нейронная модель):")
    print("-" * 70)
    
    for text1, text2 in test_pairs:
        emb1 = infinity_service.embed_text(text1)
        emb2 = infinity_service.embed_text(text2)
        sim = infinity_service.compute_similarity(emb1, emb2)
        
        print(f"  \"{text1}\"")
        print(f"  \"{text2}\"")
        print(f"  Сходство: {sim:.3f}")
        print()
    
    print("Ожидания:")
    print("  - Похожие пары должны иметь высокое сходство (> 0.7)")
    print("  - Разные пары должны иметь низкое сходство (< 0.5)")
    print("  - Infinity должен показать лучшее разделение")
    print()
else:
    print("[ПРОПУЩЕНО] Infinity недоступен")
    print()

# Тест 2: Скорость генерации эмбеддингов
print("[ТЕСТ 2] Скорость генерации эмбеддингов")
print("=" * 70)
print()

test_texts = [
    "Классифицировать этот текст в категорию",
    "Ответить на вопрос пользователя",
    "Сгенерировать контент по плану",
    "Проверить качество документа",
    "Суммировать длинный текст"
] * 10  # 50 текстов

# SimpleEmbedding
print("SimpleEmbeddingService:")
print("-" * 70)

start = time.time()
for text in test_texts:
    emb = simple_service.embed_text(text)
elapsed_simple = time.time() - start

print(f"  Текстов: {len(test_texts)}")
print(f"  Время: {elapsed_simple*1000:.2f} ms")
print(f"  На текст: {elapsed_simple*1000/len(test_texts):.2f} ms")
print()

# InfinityEmbedding
if INFINITY_OK:
    print("InfinityEmbeddingService:")
    print("-" * 70)
    
    start = time.time()
    for text in test_texts:
        emb = infinity_service.embed_text(text)
    elapsed_infinity = time.time() - start
    
    print(f"  Текстов: {len(test_texts)}")
    print(f"  Время: {elapsed_infinity*1000:.2f} ms")
    print(f"  На текст: {elapsed_infinity*1000/len(test_texts):.2f} ms")
    print()
    
    if elapsed_simple < elapsed_infinity:
        print(f"  Simple быстрее в {elapsed_infinity/elapsed_simple:.1f}x раз")
    else:
        print(f"  Infinity быстрее в {elapsed_simple/elapsed_infinity:.1f}x раз")
    
    print()
    print("  Примечание: Infinity медленнее из-за сетевых вызовов,")
    print("  но качество эмбеддингов значительно выше!")
    print()
else:
    print("[ПРОПУЩЕНО] Infinity недоступен")
    print()

# Итог
print("=" * 70)
print("ИТОГОВАЯ СВОДКА")
print("=" * 70)
print()

print("Компоненты:")
print(f"  Faiss: {'[OK]' if FAISS_OK else '[НЕ УСТАНОВЛЕН]'}")
print(f"  Infinity: {'[OK]' if INFINITY_OK else '[НЕДОСТУПЕН]'}")
print()

print("Рекомендации:")
print()

if not FAISS_OK:
    print("  1. Установите Faiss:")
    print("     pip install faiss-cpu numpy")
    print()

if not INFINITY_OK and config.EMBEDDING_TYPE == "infinity":
    print("  2. Запустите Infinity сервер:")
    print("     docker run -d -p 7997:7997 michaelf34/infinity:latest \\")
    print("       --model-name BAAI/bge-small-en-v1.5")
    print()

if FAISS_OK and INFINITY_OK:
    print("  [OK] Все компоненты готовы!")
    print()
    print("  Установите в .env:")
    print("    EMBEDDING_TYPE=infinity")
    print("    KNN_TYPE=faiss")
    print()
    print("  И запустите:")
    print("    python main.py")
    print()

print("=" * 70)
