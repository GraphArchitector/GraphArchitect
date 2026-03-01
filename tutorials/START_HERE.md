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

Вставьте ключ OpenRouter

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
- [Code Review](workflows/code_review.md) - Ревью кода
- [Research](workflows/research_workflow.md) - Исследования

**Время**: 20-30 минут на workflow  
**Результат**: Готовое решение

**Общее время**: 105 минут  

### Хочу понять как это работает

**Изучите**:
1. [Алгоритмы графа](intermediate/01_graph_algorithms.md) - 30 мин
2. [Выбор инструментов](intermediate/02_tool_selection.md) - 30 мин

**Время**: 60 минут  
**Результат**: Глубокое понимание системы

### Нужен production deployment

**Изучите**:
1. [Системы эмбеддингов](deployment/01_embedding_systems.md) - 45 мин
2. [Production Deployment](deployment/02_production_deployment.md) - 50 мин

**Время**: 95 минут  
**Результат**: Production-ready система

---

## Структура туториалов

```
tutorials/
├── START_HERE.md          # Этот файл
│
├── beginner/              # 4 туториала (60 мин)
├── intermediate/          # 2 туториала (60 мин)
├── deployment/            # 2 туториала (95 мин)
└── workflows/             # 4 сценария (105 мин)
```

---

## Что дальше?

### Главная страница

Откройте [README.md](README.md) для полного обзора.

### Затем

Выберите свой путь выше и начните обучение.

---

**Общее время**: 320 минут (~5 часов материала)  
**Уровней**: 3 (Beginner, Intermediate, Deployment)  
**Workflows**: 4 готовых сценария  

**Готовы начать**: [Quick Start](beginner/01_quick_start.md)
