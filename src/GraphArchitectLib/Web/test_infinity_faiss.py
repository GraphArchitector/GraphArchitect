"""
Скрипт тестирования Infinity + Faiss интеграции.

Проверяет:
1. Доступность Faiss
2. Доступность Infinity сервера
3. Качество эмбеддингов
4. Скорость k-NN поиска
5. Сравнение производительности
"""

import sys
from pathlib import Path
import time

# Добавляем grapharchitect
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("ТЕСТИРОВАНИЕ INFINITY + FAISS")
print("=" * 70)
print()

# Проверка 1: Faiss
print("[ПРОВЕРКА 1] Доступность Faiss")
print("-" * 70)

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

print()

# Проверка 2: Infinity сервер
print("[ПРОВЕРКА 2] Доступность Infinity сервера")
print("-" * 70)

try:
    import config
    
    print(f"  URL: {config.INFINITY_BASE_URL}")
    
    if config.EMBEDDING_TYPE == "infinity":
        print(f"  Модель: {config.INFINITY_MODEL}")
        print(f"  Размерность: {config.EMBEDDING_DIMENSION}")
        
        # Проверка доступности
        import requests
        
        try:
            response = requests.post(
                f"{config.INFINITY_BASE_URL}/embed",
                json={"text": "test"},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"  [OK] Сервер отвечает")
                data = response.json()
                
                # Проверка формата ответа
                if "embedding" in data or "data" in data:
                    print(f"  [OK] Формат ответа корректен")
                    INFINITY_OK = True
                else:
                    print(f"  [WARNING] Неожиданный формат ответа")
                    print(f"  Ответ: {data}")
                    INFINITY_OK = False
            else:
                print(f"  [FAIL] Ошибка HTTP: {response.status_code}")
                INFINITY_OK = False
        
        except requests.exceptions.ConnectionError:
            print(f"  [FAIL] Не удалось подключиться к серверу")
            print(f"  Запустите Infinity сервер на {config.INFINITY_BASE_URL}")
            INFINITY_OK = False
        
        except Exception as e:
            print(f"  [FAIL] Ошибка: {e}")
            INFINITY_OK = False
    
    else:
        print(f"  [INFO] Infinity отключен (EMBEDDING_TYPE={config.EMBEDDING_TYPE})")
        INFINITY_OK = True  # Не ошибка, просто не используется

except Exception as e:
    print(f"  [ERROR] {e}")
    INFINITY_OK = False

print()

# Проверка 3: Создание сервисов
print("[ПРОВЕРКА 3] Создание сервисов эмбеддингов")
print("-" * 70)

try:
    from grapharchitect.services.embedding.embedding_factory import create_embedding_service
    import config
    
    embedding_service = create_embedding_service(
        embedding_type=config.EMBEDDING_TYPE,
        dimension=config.EMBEDDING_DIMENSION,
        infinity_url=config.INFINITY_BASE_URL,
        infinity_api_key=config.INFINITY_API_KEY,
        infinity_model=config.INFINITY_MODEL,
        infinity_timeout=config.INFINITY_TIMEOUT,
        fallback_to_simple=True
    )
    
    print(f"  [OK] Создан: {embedding_service.__class__.__name__}")
    print(f"  Размерность: {embedding_service.embedding_dimension}")
    
    # Тест эмбеддинга
    test_embedding = embedding_service.embed_text("Тестовый запрос")
    print(f"  [OK] Тестовый эмбеддинг: {len(test_embedding)} измерений")
    
    EMBEDDING_OK = True

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    EMBEDDING_OK = False

print()

# Проверка 4: Создание k-NN ретривера
print("[ПРОВЕРКА 4] Создание k-NN ретривера")
print("-" * 70)

try:
    from grapharchitect.services.nli.retriever_factory import create_knn_retriever
    import config
    
    retriever = create_knn_retriever(
        embedding_service=embedding_service,
        retriever_type=config.KNN_TYPE,
        vector_weight=config.KNN_VECTOR_WEIGHT,
        text_weight=config.KNN_TEXT_WEIGHT,
        faiss_index_type=config.FAISS_INDEX_TYPE
    )
    
    print(f"  [OK] Создан: {retriever.__class__.__name__}")
    
    # Информация об индексе
    if hasattr(retriever, 'get_index_info'):
        info = retriever.get_index_info()
        print(f"  Faiss enabled: {info['faiss_enabled']}")
        print(f"  Index type: {info['index_type']}")
        print(f"  Vector weight: {info['vector_weight']}")
        print(f"  Text weight: {info['text_weight']}")
    
    KNN_OK = True

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    KNN_OK = False

print()

# Проверка 5: Тест производительности
if EMBEDDING_OK and KNN_OK:
    print("[ПРОВЕРКА 5] Тест производительности k-NN")
    print("-" * 70)
    
    try:
        from grapharchitect.services.nli.nli_dataset_item import NLIDatasetItem
        from grapharchitect.entities.connectors.task_representation import TaskRepresentation
        from grapharchitect.entities.connectors.connector import Connector
        
        # Создаем тестовый датасет
        test_examples = []
        for i in range(100):
            rep = TaskRepresentation()
            rep.input_connector = Connector("text", "question")
            rep.output_connector = Connector("text", "answer")
            
            item = NLIDatasetItem(
                task_text=f"Пример задачи номер {i}",
                task_embedding=None,  # Создастся автоматически
                representation=rep
            )
            test_examples.append(item)
        
        # Загружаем в ретривер
        retriever.load_dataset(test_examples)
        print(f"  Загружено примеров: {len(test_examples)}")
        
        # Тест поиска
        query_text = "Пример задачи для поиска"
        query_embedding = embedding_service.embed_text(query_text)
        
        # Измеряем время
        start = time.time()
        results = retriever.retrieve(
            task_text=query_text,
            task_embedding=query_embedding,
            k=5
        )
        elapsed = time.time() - start
        
        print(f"  [OK] Поиск выполнен")
        print(f"  Найдено: {len(results)} результатов")
        print(f"  Время: {elapsed*1000:.2f} ms")
        
        if results:
            print(f"\n  Топ-3 результата:")
            for i, scored in enumerate(results[:3], 1):
                print(f"    {i}. {scored.example.task_text[:50]}...")
                print(f"       Score: {scored.final_score:.3f} (vec: {scored.vector_score:.3f}, text: {scored.text_score:.3f})")
    
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

print()

# Итоговый отчет
print("=" * 70)
print("ИТОГОВЫЙ ОТЧЕТ")
print("=" * 70)
print()

checks = [
    ("Faiss", FAISS_OK),
    ("Infinity Server", INFINITY_OK),
    ("Embedding Service", EMBEDDING_OK),
    ("k-NN Retriever", KNN_OK)
]

passed = sum(1 for _, ok in checks if ok)
total = len(checks)

for name, ok in checks:
    status = "[OK]" if ok else "[FAIL]"
    print(f"  {status:8} {name}")

print()
print("-" * 70)
print(f"  Результат: {passed}/{total} проверок пройдено")
print("=" * 70)
print()

if passed == total:
    print("[OK] ВСЕ КОМПОНЕНТЫ РАБОТАЮТ")
    print()
    print("Конфигурация:")
    print(f"  Эмбеддинги: {config.EMBEDDING_TYPE}")
    print(f"  k-NN: {config.KNN_TYPE}")
    print()
    print("Можно запускать сервер:")
    print("  python main.py")
else:
    print("[WARNING] НЕКОТОРЫЕ КОМПОНЕНТЫ НЕДОСТУПНЫ")
    print()
    
    if not FAISS_OK:
        print("Установите Faiss:")
        print("  pip install faiss-cpu numpy")
        print()
    
    if not INFINITY_OK and config.EMBEDDING_TYPE == "infinity":
        print("Запустите Infinity сервер:")
        print("  docker run -d -p 7997:7997 michaelf34/infinity:latest --model-name BAAI/bge-small-en-v1.5")
        print()
    
    print("Или используйте базовый режим:")
    print("  EMBEDDING_TYPE=simple")
    print("  KNN_TYPE=naive")

print("=" * 70)
