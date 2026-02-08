# Workflow: Исследовательские задачи

**Сценарий**: Автоматизация исследования и анализа информации  
**Сложность**: Средняя  
**Время**: 25 минут

---

## Описание задачи

Исследовательский workflow включает:
1. Сбор информации из различных источников
2. Анализ и синтез данных
3. Выявление трендов и паттернов
4. Формулирование выводов
5. Создание исследовательского отчета

---

## Архитектура Workflow

```
Исследовательский вопрос
    ↓
┌─────────────────────────────────┐
│ Шаг 1: Information Gathering    │
│ - Поиск релевантных источников  │
│ - Извлечение ключевых фактов    │
│ - Сбор статистики               │
│ Инструменты:                    │
│   - Web Scraper                 │
│   - Trend Analyzer              │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 2: Analysis                 │
│ - Анализ собранных данных       │
│ - Выявление паттернов           │
│ - Сравнительный анализ          │
│ Инструменты:                    │
│   - Trend Analyzer              │
│   - GPT-4 Classifier            │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 3: Synthesis                │
│ - Синтез информации             │
│ - Формулирование выводов        │
│ - Выявление противоречий        │
│ Инструменты:                    │
│   - Technical Writer            │
│   - Formal Responder            │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 4: Report Generation        │
│ - Структурирование отчета       │
│ - Добавление визуализаций       │
│ - Форматирование                │
│ Инструменты:                    │
│   - Summary Reporter            │
│   - Technical Writer            │
└────────────┬────────────────────┘
             ↓
Исследовательский отчет
```

---

## Реализация

### Базовое исследование

```python
import requests
import json

def research_topic(question):
    """
    Провести исследование по вопросу.
    
    Args:
        question: Исследовательский вопрос
        
    Returns:
        Исследовательский отчет
    """
    message = f"""
    Провести исследование по вопросу: {question}
    
    Необходимо:
    1. Найти ключевые источники и факты
    2. Проанализировать текущее состояние
    3. Выявить тренды и направления
    4. Сформулировать выводы
    5. Создать структурированный отчет
    """
    
    response = requests.post(
        "http://localhost:8000/api/chat/research/message/stream",
        data={
            "message": message,
            "planning_algorithm": "yen_10"  # Больше альтернатив для исследования
        },
        stream=True
    )
    
    sections = {
        "executive_summary": "",
        "findings": "",
        "analysis": "",
        "conclusions": ""
    }
    
    current_section = None
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            
            if chunk['type'] == 'text':
                content = chunk['content']
                
                # Определяем секцию
                if "Executive Summary" in content:
                    current_section = "executive_summary"
                elif "Findings" in content:
                    current_section = "findings"
                elif "Analysis" in content:
                    current_section = "analysis"
                elif "Conclusions" in content:
                    current_section = "conclusions"
                
                if current_section:
                    sections[current_section] += content
    
    return sections


# Использование
research = research_topic(
    "Какое влияние оказывает искусственный интеллект на рынок труда?"
)

print("Executive Summary:")
print(research['executive_summary'])

print("\nKey Findings:")
print(research['findings'])
```

---

## Типы исследований

### 1. Обзор литературы (Literature Review)

```python
message = """
Провести обзор литературы по теме: {topic}

Требования:
- Найти ключевые публикации (последние 3 года)
- Выявить основные направления исследований
- Определить пробелы в знаниях
- Предложить направления для будущих исследований
"""
```

**Workflow**:
```
[Web Scraper] → [Trend Analyzer] → [Technical Writer] → [Summary Reporter]
```

### 2. Сравнительный анализ (Comparative Analysis)

```python
message = """
Сравнить {option_A} и {option_B}:

Критерии:
- Функциональность
- Производительность
- Стоимость
- Ease of use
- Ecosystem

Вывод: Какой вариант лучше для {use_case}
"""
```

**Workflow**:
```
[Web Scraper] → [Trend Analyzer] → [Structured Outliner] → [Technical Writer] → [QA]
```

### 3. Анализ трендов (Trend Analysis)

```python
message = """
Проанализировать тренды в {domain} за последние {period}:

- Основные тренды
- Темпы роста
- Драйверы изменений
- Прогноз на будущее
"""
```

**Workflow**:
```
[Web Scraper] → [Trend Analyzer] → [Summary Reporter]
```

---

## Интеграция с источниками данных

### Поиск в интернете

```python
def research_with_web_sources(topic):
    """Исследование с использованием веб-источников"""
    
    # Шаг 1: Сбор данных
    sources = [
        "https://scholar.google.com/search?q=" + topic,
        "https://arxiv.org/search/?query=" + topic,
        "https://www.semanticscholar.org/search?q=" + topic
    ]
    
    # В реальном инструменте Web Scraper
    # Сейчас через prompt
    message = f"""
    Исследовать тему: {topic}
    
    Источники для анализа:
    {chr(10).join(sources)}
    
    Найти:
    - Последние публикации
    - Ключевые авторы
    - Основные выводы
    """
    
    # Отправка на GraphArchitect
    # ...
```

### Работа с документами

```python
# Загрузка PDF исследований
files = {
    'file1': open('paper1.pdf', 'rb'),
    'file2': open('paper2.pdf', 'rb'),
    'file3': open('paper3.pdf', 'rb')
}

for name, file in files.items():
    requests.post(
        "http://localhost:8000/api/chat/research/document",
        files={'file': file}
    )

# Анализ загруженных документов
requests.post(
    "http://localhost:8000/api/chat/research/message/stream",
    data={
        "message": "Проанализировать загруженные исследования и создать обзор"
    }
)
```

---

## Качество исследования

### Критерии

| Критерий | Как проверить |
|----------|---------------|
| Полнота | Покрыты все аспекты вопроса? |
| Актуальность | Используются свежие данные? |
| Объективность | Рассмотрены разные точки зрения? |
| Глубина | Достаточно деталей? |
| Структура | Логичная организация? |
| Цитирование | Указаны источники? |

### Автоматическая проверка

```python
# QA инструмент проверяет:
qa_criteria = {
    "completeness": 0.85,      # Полнота
    "relevance": 0.90,         # Актуальность
    "objectivity": 0.80,       # Объективность
    "depth": 0.75,             # Глубина
    "structure": 0.95,         # Структура
    "citations": 0.70          # Цитирование
}

overall_score = sum(qa_criteria.values()) / len(qa_criteria)
print(f"Overall research quality: {overall_score:.1%}")
```

---

## Примеры исследовательских вопросов

### Технические исследования

```
1. "Сравнение архитектур нейронных сетей для NLP задач"
2. "Обзор методов оптимизации производительности Python"
3. "Анализ security практик в микросервисных архитектурах"
```

### Бизнес-исследования

```
1. "Анализ рынка электромобилей в 2026 году"
2. "Тренды в B2B SaaS продуктах"
3. "Влияние удаленной работы на продуктивность"
```

### Академические исследования

```
1. "Обзор подходов к explainable AI"
2. "Современные методы recommendation systems"
3. "Этические вопросы использования AI в медицине"
```

---

## Масштабирование

### Batch исследования

```python
questions = [
    "Тренды в AI/ML 2026",
    "Blockchain в финтех",
    "Квантовые вычисления",
    "Edge computing",
    "Web3 технологии"
]

reports = []
for question in questions:
    report = research_topic(question)
    reports.append(report)
    print(f"[OK] Completed: {question}")

# Мета-анализ всех отчетов
meta_analysis = analyze_multiple_reports(reports)
```

---

## Итоги

### Вы создали

- Workflow для автоматизации исследований
- 4-шаговый процесс (сбор → анализ → синтез → отчет)
- Систему контроля качества
- Интеграцию с внешними источниками

### Применение

- Обзоры литературы
- Сравнительный анализ
- Анализ трендов
- Due diligence
- Конкурентный анализ

---

**Следующий workflow**: [Document Processing](document_processing.md)
