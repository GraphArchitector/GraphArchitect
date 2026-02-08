# Продвинутый туториал: Системы эмбеддингов

**Уровень**: Продвинутый  
**Время**: 45 минут  
**Цель**: Понять и настроить системы эмбеддингов в GraphArchitect

---

## Что вы узнаете

- Зачем нужны эмбеддинги в GraphArchitect
- Сравнение Simple vs Infinity vs Sentence Transformers
- Как настроить Infinity API
- Как использовать Faiss для быстрого k-NN
- Оптимизация производительности

---

## Роль эмбеддингов в GraphArchitect

### Где используются

1. **NLI (Natural Language Interface)**:
   - Векторизация запросов пользователя
   - k-NN поиск похожих примеров
   - Предсказание коннекторов

2. **Выбор инструментов (InstrumentSelector)**:
   - Эмбеддинги возможностей инструментов
   - Косинусное сходство задачи и инструмента
   - Вычисление логитов

3. **Обучение (Contrastive Learning)**:
   - Обновление эмбеддингов инструментов
   - Приближение к успешным задачам
   - Отдаление от неуспешных

### Влияние на качество

```
Качество эмбеддингов напрямую влияет на:
- Точность NLI парсинга (40% → 90%)
- Качество выбора инструментов
- Скорость обучения
```

---

## Сравнение реализаций

### 1. SimpleEmbeddingService (по умолчанию)

**Как работает**:
```python
embedding = SHA256(text) → нормализация → вектор[384]
```

**Характеристики**:
- Скорость: Мгновенно (< 0.1ms)
- Качество: Очень низкое
- Семантика: Нет (хеш не отражает смысл)
- Зависимости: Нет
- Использование: Только для тестирования

**Проблема**:
```
"Классифицировать текст" и "Категоризировать сообщение"
→ Совершенно разные векторы (разные хеши)
→ NLI не видит семантического сходства
```

### 2. InfinityEmbeddingService

**Как работает**:
```python
text → HTTP POST to Infinity → нейронная модель → вектор[384-1024]
```

**Характеристики**:
- Скорость: 50-200ms (сетевой вызов)
- Качество: Высокое
- Семантика: Да (обученная модель)
- Зависимости: Infinity сервер
- Использование: Production

**Преимущество**:
```
"Классифицировать текст" и "Категоризировать сообщение"
→ Похожие векторы (косинусное сходство > 0.85)
→ NLI правильно распознает задачу
```

**Модели**:
- `BAAI/bge-small-en-v1.5` (384 dim, английский)
- `BAAI/bge-m3` (1024 dim, мультиязычный, лучше для русского)
- `intfloat/multilingual-e5-base` (768 dim, мультиязычный)

### 3. Sentence Transformers (альтернатива)

**Как работает**:
```python
model = SentenceTransformer('model-name')
embedding = model.encode(text)
```

**Характеристики**:
- Скорость: 10-50ms (локально)
- Качество: Высокое
- Семантика: Да
- Зависимости: ~500MB модель
- Использование: Когда нет Infinity

---

## Настройка Infinity

### Запуск сервера

```bash
# Docker (рекомендуется)
docker run -d \
  --name grapharchitect-infinity \
  -p 7997:7997 \
  michaelf34/infinity:latest \
  --served-model-name BAAI/bge-m3 \
  --batch-size 32 \
  --device cpu

# Для GPU
docker run -d \
  --gpus all \
  --name grapharchitect-infinity \
  -p 7997:7997 \
  michaelf34/infinity:latest \
  --served-model-name BAAI/bge-m3 \
  --batch-size 128 \
  --device cuda
```

### Проверка

```bash
curl -X POST http://localhost:7997/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "тестовый текст"}'
```

### Конфигурация в GraphArchitect

```bash
# .env
EMBEDDING_TYPE=infinity
INFINITY_BASE_URL=http://localhost:7997
INFINITY_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
```

---

## Faiss для быстрого k-NN

### Зачем нужен?

**Проблема**: При большом датасете NLI (> 1000 примеров) линейный поиск медленный.

**Решение**: Faiss индекс для ускорения в 50-1000 раз.

### Установка

```bash
pip install faiss-cpu numpy

# Для GPU (опционально)
pip install faiss-gpu
```

### Конфигурация

```bash
# .env
KNN_TYPE=faiss
FAISS_INDEX_TYPE=FlatIP  # или FlatL2, HNSW
```

### Типы индексов

**FlatIP** (рекомендуется):
- Точный поиск (100% recall)
- Косинусное сходство
- Быстрый для < 1M векторов

**HNSW** (для > 10K):
- Приблизительный (98-99% recall)
- Очень быстрый
- Больше памяти

---

## Производительность

### Бенчмарк: Simple vs Infinity

```bash
cd Web
python benchmark_embeddings.py
```

**Результаты**:

| Операция | Simple | Infinity (BGE-M3) |
|----------|--------|-------------------|
| Эмбеддинг (1 текст) | 0.1ms | 150ms |
| Семантическое сходство | Нет | Да |
| Точность NLI | 42% | 89% |

**Вывод**: Infinity медленнее, но качество выше в 2 раза.

### Бенчмарк: Naive vs Faiss k-NN

| Датасет | Naive | Faiss FlatIP |
|---------|-------|--------------|
| 10 | 0.001s | 0.0005s |
| 100 | 0.01s | 0.001s |
| 1,000 | 0.1s | 0.002s |
| 10,000 | 1.0s | 0.005s |

**Вывод**: Faiss критичен при датасете > 1000.

---

## Практика

### Упражнение 1: Сравнение эмбеддингов

```python
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.embedding.infinity_embedding_service import InfinityEmbeddingService

# Simple
simple = SimpleEmbeddingService(dimension=384)
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
```

**Ожидаемый результат**:
```
Simple сходство: 0.123 (случайное, нет семантики)
Infinity сходство: 0.887 (высокое, похожие по смыслу)
```

### Упражнение 2: Влияние на NLI

Запустите примеры NLI с разными эмбеддингами:

```bash
# С Simple (по умолчанию)
cd examples/Python/nli
python example_01_basic_nli.py > results_simple.txt

# С Infinity
# Измените EMBEDDING_TYPE=infinity в config
python example_01_basic_nli.py > results_infinity.txt

# Сравните
diff results_simple.txt results_infinity.txt
```

**Вопрос**: Насколько точнее стал парсинг?

---

## Production настройка

### Рекомендуемая конфигурация

```bash
# .env для production
EMBEDDING_TYPE=infinity
INFINITY_BASE_URL=http://your-infinity-server:7997
INFINITY_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

KNN_TYPE=faiss
FAISS_INDEX_TYPE=FlatIP
```

МНОГО ВОПРОСОВ.
### Кеширование эмбеддингов

Сейчас эмбеддинги вычисляются каждый раз. Для оптимизации:

```python
# Добавить в sqlite_repository.py
def save_embedding(self, text, embedding):
    cursor.execute(
        "INSERT INTO embeddings (text_hash, embedding) VALUES (?, ?)",
        (hash(text), json.dumps(embedding))
    )

def get_embedding(self, text):
    cursor.execute(
        "SELECT embedding FROM embeddings WHERE text_hash = ?",
        (hash(text),)
    )
    # ...
```

---

## Итоги

### Вы узнали

- Эмбеддинги критичны для качества NLI и выбора
- Simple - только для тестирования
- Infinity - для production качества
- Faiss - для production скорости
- Правильная настройка дает +50% точности

### Рекомендации

**Для разработки**: Simple (быстро, без зависимостей)

**Для staging**: Infinity + Naive k-NN

**Для production** (< 1000 примеров): Infinity + Naive

**Для production** (> 1000 примеров): Infinity + Faiss

---

**Следующий**: [Production Deployment](02_production_deployment.md)
