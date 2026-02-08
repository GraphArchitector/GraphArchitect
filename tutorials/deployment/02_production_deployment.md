# Туториал: Развертывание моделей

**Уровень**: Продвинутый  
**Время**: 30 минут  
**Цель**: Научиться разворачивать модели для GraphArchitect

---

## Что вы узнаете

- Как развернуть Infinity сервер для эмбеддингов
- Как развернуть VLLM сервер для LLM
- Как использовать Qwen NLI модель
- Как настроить GraphArchitect для работы с моделями

---

## Модель 1: Infinity (Эмбеддинги)

### Назначение

Infinity предоставляет высококачественные семантические эмбеддинги через API.

**Используется в**:
- NLI (парсинг естественного языка)
- Выбор инструментов (косинусное сходство)
- Обучение (Contrastive Learning)

### Развертывание через Docker

```bash
# Базовая команда (CPU)
docker run -d \
  --name grapharchitect-infinity \
  -p 7997:7997 \
  michaelf34/infinity:latest \
  --model-name BAAI/bge-m3 \
  --batch-size 32 \
  --device cpu

# С GPU (рекомендуется)
docker run -d \
  --gpus all \
  --name grapharchitect-infinity \
  -p 7997:7997 \
  michaelf34/infinity:latest \
  --model-name BAAI/bge-m3 \
  --batch-size 128 \
  --device cuda
```

### Проверка работы

```bash
# Health check
curl http://localhost:7997/health

# Тест эмбеддинга
curl -X POST http://localhost:7997/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "тестовый текст"}'
```

### Рекомендуемые модели

#### Мультиязычные (русский + английский)

```bash
# BGE-M3 (лучший для русского)
--model-name BAAI/bge-m3
# Размерность: 1024

# E5 Multilingual
--model-name intfloat/multilingual-e5-base
# Размерность: 768
```

#### Только английский

```bash
# BGE Small (быстрый)
--model-name BAAI/bge-small-en-v1.5
# Размерность: 384

# E5 Base (качественный)
--model-name intfloat/e5-base-v2
# Размерность: 768
```

### Конфигурация GraphArchitect

```bash
# В .env
EMBEDDING_TYPE=infinity
INFINITY_BASE_URL=http://localhost:7997
INFINITY_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
```

---

## Модель 2: VLLM (LLM сервер)

### Назначение

VLLM предоставляет быстрый inference для языковых моделей.

**Используется в**:
- LLM Critic (RLAIF оценка)
- Инструменты (выполнение задач)
- Qwen NLI (парсинг)

### Развертывание через Docker

```bash
# Qwen 2.5 7B
docker run -d \
  --gpus all \
  --name grapharchitect-vllm \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype auto \
  --max-model-len 4096

# Для CPU (медленно, не рекомендуется)
docker run -d \
  --name grapharchitect-vllm \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cpu
```

### Проверка работы

```bash
# Health check
curl http://localhost:8000/health

# Тест генерации
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "Привет, как дела?",
    "max_tokens": 100
  }'
```

### Конфигурация для RLAIF

```python
# В config.py
RLAIF_BACKEND = "vllm"
VLLM_HOST = "http://localhost:8000"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

---

## Модель 3: Qwen NLI (Fine-tuned)

### Назначение

Дообученная Qwen для точного парсинга задач в коннекторы.

### Получение модели

**Из отчета**: Модель `qwen-nli-7b` была дообучена

**Где получить**:
1. Из архива проекта (если есть)
2. Дообучить самостоятельно (см. раздел ниже)
3. Запросить у авторов отчета

### Развертывание

```bash
# Через Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/qwen-nli-7b",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("/path/to/qwen-nli-7b")
```

### Использование в GraphArchitect

```python
from grapharchitect.services.nli.qwen_nli_service import QwenNLIService

qwen_nli = QwenNLIService(
    model_path="/path/to/qwen-nli-7b",
    device="cuda"
)

# В grapharchitect_bridge.py
if qwen_nli.is_available():
    self.nli = qwen_nli  # Вместо k-NN
```

---

## Модель 4: E5 для k-NN (опционально)

### Назначение

Sentence Transformers локально (альтернатива Infinity).

### Установка

```bash
pip install sentence-transformers
```

### Использование

```python
from sentence_transformers import SentenceTransformer

class E5EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('intfloat/e5-base-v2')
    
    def embed_text(self, text):
        return self.model.encode(text).tolist()
```

---

## Полная конфигурация для всех моделей

### docker-compose.yml

```yaml
version: '3.8'

services:
  # Infinity для эмбеддингов
  infinity:
    image: michaelf34/infinity:latest
    ports:
      - "7997:7997"
    command: >
      --model-name BAAI/bge-m3
      --batch-size 128
      --device cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # VLLM для LLM
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --dtype auto
      --max-model-len 4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # GraphArchitect Web API
  grapharchitect:
    build: ./src/GraphArchitectLib/Web
    ports:
      - "8080:8000"
    environment:
      - EMBEDDING_TYPE=infinity
      - INFINITY_BASE_URL=http://infinity:7997
      - RLAIF_BACKEND=vllm
      - VLLM_HOST=http://vllm:8000
    depends_on:
      - infinity
      - vllm
```

### Запуск

```bash
docker-compose up -d
```

---

## Требования к железу

### Минимальные (CPU)

- **RAM**: 16 GB
- **CPU**: 4+ cores
- **Диск**: 20 GB
- **Скорость**: Медленно, только для тестов

### Рекомендуемые (GPU)

- **GPU**: NVIDIA с 8+ GB VRAM
- **RAM**: 32 GB
- **CPU**: 8+ cores
- **Диск**: 50 GB
- **Скорость**: Производительно

### Production

- **GPU**: NVIDIA A100/H100 или несколько A10/T4
- **RAM**: 64+ GB
- **CPU**: 16+ cores
- **Диск**: 100+ GB SSD

---

## Проверка развертывания

### Чеклист

```bash
# 1. Infinity работает
curl http://localhost:7997/health
# Ожидается: 200 OK

# 2. VLLM работает
curl http://localhost:8000/health
# Ожидается: 200 OK

# 3. GraphArchitect подключается
cd src/GraphArchitectLib/Web
python test_infinity_faiss.py
# Ожидается: все проверки [OK]

# 4. RLAIF работает
cd examples/Python/rlaif
python example_llm_critic.py
# Ожидается: оценки от LLM

# 5. Web API работает
python main.py
# Логи: "InfinityEmbeddingService initialized"
```

---

## Troubleshooting

### Infinity не запускается

**Проблема**: OOM (Out of Memory)

**Решение**: Уменьшите batch_size:
```bash
--batch-size 16  # Вместо 128
```

### VLLM ошибка

**Проблема**: CUDA out of memory

**Решение**: Используйте меньшую модель:
```bash
--model Qwen/Qwen2.5-1.5B-Instruct  # Вместо 7B
```

### Qwen NLI не загружается

**Проблема**: Модель не найдена

**Решение**: Проверьте путь или используйте k-NN:
```python
# В config.py
NLI_TYPE = "knn"  # Вместо "qwen"
```

---

## Итоги

### Вы научились

- Разворачивать Infinity для эмбеддингов
- Разворачивать VLLM для LLM
- Конфигурировать GraphArchitect для моделей
- Проверять работоспособность

### Следующие шаги

1. Протестируйте каждую модель отдельно
2. Запустите GraphArchitect с моделями
3. Измерьте улучшение качества
4. Оптимизируйте под вашу нагрузку

---

**Следующий туториал**: [Оптимизация производительности](../advanced/02_performance_tuning.md)
