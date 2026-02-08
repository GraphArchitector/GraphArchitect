# Начните здесь

Добро пожаловать в туториалы GraphArchitect!

---

## Что такое GraphArchitect?

**GraphArchitect** - это система планирования и выполнения задач на основе графа инструментов с:

- Алгоритмами поиска путей (Dijkstra, A*, Yen, ACO)
- Вероятностным выбором инструментов (Softmax)
- Автоматическим обучением (Policy Gradient)
- Natural Language Interface (NLI)
- Web API и SQLite базой данных

---

## Быстрый старт (5 минут)

### 1. Инициализация

```bash
cd .\src\GraphArchitectLib\Web
python db_manager.py init
python db_manager.py load_agents
```

### 2. Запуск

```bash
python main.py
```

### 3. Использование

Откройте: `http://localhost:8000`

Отправьте: `"Классифицировать этот текст"`

---

## Выберите свой путь

### Я новичок

**Начните с**:
1. [Быстрый старт](beginner/01_quick_start.md) - 5 мин
2. [Основные концепции](beginner/02_basic_concepts.md) - 15 мин
3. [Первый Workflow](beginner/03_first_workflow.md) - 20 мин

**Время**: 40 минут  
**Результат**: Понимание основ и первый workflow

### Мне нужно решить задачу

**Выберите workflow**:
- [Customer Support](workflows/customer_support.md) - Обработка запросов
- [Content Creation](workflows/content_creation.md) - Создание контента
- [Data Analysis](workflows/data_analysis.md) - Анализ данных
- [Code Review](workflows/code_review.md) - Ревью кода
- [Research](workflows/research_workflow.md) - Исследования
- [Documents](workflows/document_processing.md) - Обработка документов

**Время**: 20-30 минут на workflow  
**Результат**: Готовое решение

### Хочу понять как это работает

**Изучите**:
1. [Алгоритмы графа](intermediate/01_graph_algorithms.md) - 30 мин
2. [Выбор инструментов](intermediate/02_tool_selection.md) - 30 мин
3. [Система обучения](intermediate/04_training_system.md) - 30 мин

**Время**: 90 минут  
**Результат**: Глубокое понимание системы

### Нужен production deployment

**Изучите**:
1. [Системы эмбеддингов](deployment/01_embedding_systems.md) - 45 мин
2. [Production Deployment](deployment/02_production_deployment.md) - 50 мин
3. [Performance Tuning](advanced/02_performance_tuning.md) - 35 мин

**Время**: 130 минут  
**Результат**: Production-ready система

---

## Структура туториалов

```
tutorials/
├── README.md              # Обзор
├── INDEX.md               # Полный индекс
├── START_HERE.md          # Этот файл
├── TUTORIAL_GUIDE.md      # Руководство
├── QUICK_REFERENCE.md     # Справочник
│
├── beginner/              # 5 туториалов (75 мин)
├── intermediate/          # 5 туториалов (150 мин)
├── advanced/              # 5 туториалов (200 мин)
└── workflows/             # 6 сценариев (150 мин)
```

---

## Что дальше?

### Сейчас

Откройте [README.md](README.md) для полного обзора.

### Затем

Выберите свой путь выше и начните обучение.

### Вопросы?

Смотрите [TUTORIAL_GUIDE.md](TUTORIAL_GUIDE.md) для помощи.

---

**Общее время**: 575 минут (~10 часов материала)  
**Уровней**: 3 (Beginner, Intermediate, Advanced)  
**Workflows**: 6 готовых сценариев  
**Примеров**: 50+

**Готовы начать**: [README.md](README.md) → [Quick Start](beginner/01_quick_start.md)
