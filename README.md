![Stars](https://img.shields.io/github/stars/GraphArchitector/GraphArchitect?style=flat-square)
![Forks](https://img.shields.io/github/forks/GraphArchitector/GraphArchitect?style=flat-square)
![Watchers](https://img.shields.io/github/watchers/GraphArchitector/GraphArchitect?style=flat-square)


<img src="https://github.com/GraphArchitector/GraphArchitect/blob/main/docs/img/logo_ga.png?raw=true" width=400 />


# GraphArchitect

Открытая библиотека для планирования и выполнения многошаговых задач с помощью графа инструментов.

---

## Описание

**GraphArchitect** (Графовый архитектор) - система автоматического планирования и выполнения задач на основе графа инструментов с обучением через Policy Gradient.

### Ключевые возможности

- **Граф-планирование**: Dijkstra, A*, Yen, ACO алгоритмы
- **Интеллектуальный выбор**: Softmax с адаптивной температурой
- **Автообучение**: Policy Gradient + Contrastive Learning
- **Natural Language**: NLI для парсинга естественного языка
- **Интеграции**: LangChain, A2A, MCP протоколы
- **Web API**: FastAPI + WebSocket + SQLite

---

## Быстрый старт

```bash
cd src/GraphArchitectLib/Web

# Инициализация
python db_manager.py init
python db_manager.py load_agents

# Запуск
python main.py

# Открыть
http://localhost:8000
```

---

## Структура проекта

```
GraphArchitect/
├── src/GraphArchitectLib/
│   ├── grapharchitect/          # Основная библиотека
│   │   ├── algorithms/          # Dijkstra, A*, Yen, ACO
│   │   ├── entities/            # BaseTool, Connector
│   │   ├── services/            # NLI, Selection, Training
│   │   ├── tools/               # API интеграции
│   │   └── protocols/           # A2A, MCP
│   │
│   ├── Web/                     # Web API
│   │   ├── main.py             # FastAPI app
│   │   ├── api_router.py       # 16 endpoints
│   │   └── grapharchitect_bridge.py
│   │
│   └── Tests/                  
│
├── tutorials/                   # 21 туториал
│   ├── beginner/               # 5 туториалов
│   ├── intermediate/           # 5 туториалов
│   ├── advanced/               # 5 туториалов
│   └── workflows/              # 6 сценариев
│
├── integrations/                # Интеграции
│   └── langchain/              # LangChain
│
└── examples/                    # Примеры кода
    └── Python/
       ├── nli/                # NLI примеры
       ├── advanced/           # Продвинутые
       └── protocols/          # A2A/MCP

```

---

## Возможности

### Планирование задач

4 алгоритма поиска путей:
- **Dijkstra**: Один кратчайший путь (быстро)
- **A***: С эвристикой (еще быстрее)
- **Yen**: Топ-K путей (альтернативы)
- **ACO**: Муравьиный (разнообразие)

### Выбор инструментов

2 метода:
- **Simple**: Базовый (логиты + softmax)
- **Advanced**: С формулой R(x) (качество + стоимость + время)

### Natural Language Interface

2 подхода:
- **k-NN few-shot**: Всегда работает
- **Qwen fine-tuned**: Высокая точность (требуется модель)

### Обучение

- **Policy Gradient**: Обновление репутации
- **Contrastive Learning**: Обновление эмбеддингов
- **Автоматическое**: После каждого выполнения

---

## Интеграции

### LangChain

- GraphArchitect tools → LangChain
- LangChain tools → GraphArchitect
- Hybrid Executor

### Протоколы

- **A2A** (Agent2Agent) - Google
- **MCP** (Model Context Protocol) - Anthropic

---

## Модели и датасет

### Эмбеддеры (NLI (k-NN)/Выбор инструментов)

- [FractalGPT/E5SmallDistilV2](https://huggingface.co/FractalGPT/E5SmallDistilV2) — модель обученная на задаче NLI на базе E5Small, (100 млн параметров) 
- [FractalGPT/SbertDistilV2](https://huggingface.co/FractalGPT/SbertDistilV2) — модель обученная на задаче NLI на базе [FractalGPT/SbertDistil](https://huggingface.co/FractalGPT/SbertDistil), (11.8 млн параметров) 

### LLM для NLI
- [Ponimash/Qwen2.5-nli-7b](https://huggingface.co/Ponimash/Qwen2.5-nli-7b) — модель на базе [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) для преобразования задач на естественном языке в язык MASL.   [Пример на Colab](https://colab.research.google.com/drive/1RI3d0W-KxlSfoXI2yjm-jMA5PnuSFulV)

### NLI Датасет
- [Ponimash/nli_dataset](https://huggingface.co/datasets/Ponimash/nli_dataset) — датасет перевода задач (9 999 примеров)
---

## Документация

### Для начинающих

- [Quick Start](tutorials/beginner/01_quick_start.md) - 5 минут
- [Basic Concepts](tutorials/beginner/02_basic_concepts.md) - 15 минут
- [First Workflow](tutorials/beginner/03_first_workflow.md) - 20 минут

### Для разработчиков

- [Graph Algorithms](tutorials/intermediate/01_graph_algorithms.md)
- [Tool Selection](tutorials/intermediate/02_tool_selection.md)
- [Custom Tools](tutorials/intermediate/05_custom_tools.md)

### Готовые сценарии

- [Customer Support](tutorials/workflows/customer_support.md)
- [Content Creation](tutorials/workflows/content_creation.md)
- [Data Analysis](tutorials/workflows/data_analysis.md)
- [Code Review](tutorials/workflows/code_review.md)

### Для production

- [Embedding Systems](tutorials/advanced/01_embedding_systems.md)
- [Production Deployment](tutorials/deployment/02_production_deployment.md)
- [Performance Tuning](tutorials/advanced/02_performance_tuning.md)

---

## API

### REST Endpoints (16)

```bash
# Workflow
POST /api/chat/{id}/workflow
POST /api/chat/{id}/message/stream

# Training
POST /api/training/feedback
GET  /api/training/statistics

# Health
GET  /api/health
```

Полная документация (API): `http://localhost:8000/docs`

---

## Конфигурация

```bash
# .env
EMBEDDING_TYPE=infinity
INFINITY_BASE_URL=http://localhost:7997

KNN_TYPE=faiss
SELECTOR_TYPE=advanced

NLI_TYPE=qwen
QWEN_MODEL_PATH=/path/to/qwen-nli-7b
```

---

## Требования

### Минимальные

- Python 3.8+
- SQLite3 (встроен)

### Рекомендуемые

- Python 3.10+
- faiss-cpu (для скорости)
- Infinity server (для качества)

### Опциональные

- Transformers (для NLI)
- LangChain (для интеграции)
- Docker (для deployment)

---

## Установка

```bash
# Клонирование
git clone <repo-url>
cd GraphArchitect

# Зависимости
cd src/GraphArchitectLib/Web
pip install -r requirements.txt

# Инициализация
python db_manager.py init
python db_manager.py load_agents

# Запуск
python main.py
```

---

## Примеры

### Python API

```python
from grapharchitect.services.execution.execution_orchestrator import ExecutionOrchestrator

orchestrator = ExecutionOrchestrator(...)
context = orchestrator.execute_task(task, tools)

print(f"Результат: {context.result}")
```

### REST API

```bash
curl -X POST http://localhost:8000/api/chat/demo/message/stream \
  -F "message=Классифицировать этот текст" \
  -F "planning_algorithm=yen_5"
```

### Web Interface

Откройте: `http://localhost:8000`

---


## Научная база

Проект основан на научно-техническом отчете по НИОКР "Разработка открытой библиотеки Графовый Архитектор".

**Ключевые концепции**:
- Граф коннекторов для планирования
- Обучаемые функции качества
- Softmax с адаптивной температурой
- Policy Gradient обучение

---

## Команда

- Носко В.И. - Стандарты коннекторов
- Понимаш З.А. - Алгоритмы и обучение
- Потанин М.В. - NLI и документация
- Рудаков И.С. - База данных и датасет

---

## Лицензия

[Apache 2.0](https://github.com/GraphArchitector/GraphArchitect/blob/main/LICENSE)

---

## Ссылки

- **Туториалы**: [tutorials/START_HERE.md](tutorials/START_HERE.md)
- **Web API**: [src/GraphArchitectLib/Web/README_FINAL.md](src/GraphArchitectLib/Web/README_FINAL.md)
- **Интеграции**: [integrations/README.md](integrations/README.md)
- **Примеры**: [examples/](examples/)

---

**Начните с**: [tutorials/START_HERE.md](tutorials/START_HERE.md)
