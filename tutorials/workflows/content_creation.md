# Workflow: Создание контента

**Сценарий**: Автоматическое создание статей и контента  
**Сложность**: Средняя  
**Время**: 25 минут

---

## Описание задачи

Создание качественного контента проходит несколько этапов:
1. Исследование темы и сбор информации
2. Создание структурированного плана
3. Написание контента на основе плана
4. Проверка стиля и улучшение
5. Финальный контроль качества

---

## Архитектура Workflow

```
Тема статьи
    ↓
┌─────────────────────────────────┐
│ Шаг 1: Research                 │
│ - Поиск информации по теме      │
│ - Анализ трендов                │
│ - Сбор ключевых фактов          │
│ Инструменты:                    │
│   - Trend Analyzer              │
│   - Web Scraper                 │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 2: Outlining                │
│ - Создание структуры            │
│ - Определение разделов          │
│ - План аргументации             │
│ Инструменты:                    │
│   - Structured Outliner         │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 3: Writing                  │
│ - Написание текста              │
│ - Раскрытие каждого раздела     │
│ - Добавление примеров           │
│ Инструменты:                    │
│   - Technical Writer            │
│   - Creative Responder          │
│   - Formal Responder            │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 4: Style Improvement        │
│ - Проверка грамматики           │
│ - Улучшение стиля               │
│ - Оптимизация формулировок      │
│ Инструменты:                    │
│   - Style Checker               │
│   - Style Improver              │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 5: Quality Assurance        │
│ - Проверка фактов               │
│ - Проверка структуры            │
│ - Финальная валидация           │
│ Инструменты:                    │
│   - Strict QA                   │
└────────────┬────────────────────┘
             ↓
Готовая статья
```

---

## Реализация

### Простой вариант (3 шага)

```python
import requests

def create_article_simple(topic):
    """
    Упрощенный workflow создания статьи.
    
    План → Написание → Проверка
    """
    response = requests.post(
        "http://localhost:8000/api/chat/content/message/stream",
        data={
            "message": f"Создать статью на тему: {topic}",
            "planning_algorithm": "yen_5"
        },
        stream=True
    )
    
    result = ""
    for line in response.iter_lines():
        if line:
            import json
            chunk = json.loads(line)
            
            if chunk['type'] == 'text':
                result += chunk['content']
    
    return result


# Использование
article = create_article_simple("Машинное обучение в 2026 году")
print(article)
```

### Полный вариант (5 шагов)

```python
def create_article_full(topic, target_audience="general", tone="professional"):
    """
    Полный workflow с настройкой под аудиторию.
    """
    # Шаг 1: Исследование
    research_request = {
        "message": f"Исследовать тему: {topic}. Найти ключевые тренды и факты.",
        "planning_algorithm": "yen_5"
    }
    
    # Шаг 2: Создание плана
    outline_request = {
        "message": f"Создать структурированный план статьи о: {topic}. "
                   f"Аудитория: {target_audience}, Тон: {tone}",
        "planning_algorithm": "yen_5"
    }
    
    # Шаг 3: Написание
    writing_request = {
        "message": f"Написать статью на тему: {topic} по плану. "
                   f"Тон: {tone}",
        "planning_algorithm": "yen_5"
    }
    
    # Шаг 4: Улучшение стиля
    style_request = {
        "message": f"Улучшить стиль и грамматику статьи",
        "planning_algorithm": "yen_5"
    }
    
    # Шаг 5: Контроль качества
    qa_request = {
        "message": f"Проверить качество и валидировать статью",
        "planning_algorithm": "yen_5"
    }
    
    # Выполнение каждого шага
    results = {}
    for step_name, request in [
        ("research", research_request),
        ("outline", outline_request),
        ("writing", writing_request),
        ("style", style_request),
        ("qa", qa_request)
    ]:
        response = requests.post(
            f"http://localhost:8000/api/chat/content-{step_name}/message/stream",
            data=request,
            stream=True
        )
        
        result = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if chunk['type'] == 'text':
                    result += chunk['content']
        
        results[step_name] = result
        print(f"[OK] {step_name} completed")
    
    return results


# Использование
article = create_article_full(
    topic="Искусственный интеллект в медицине",
    target_audience="medical_professionals",
    tone="technical"
)
```

---

## Типы контента

### Технические статьи

```
[Research] → [Technical Writer] → [Style Checker] → [Strict QA]
```

Фокус на точности и технических деталях.

### Маркетинговый контент

```
[Trend Analyzer] → [Creative Responder] → [Style Improver] → [Balanced QA]
```

Фокус на креативности и вовлеченности.

### Образовательный контент

```
[Research] → [Structured Outliner] → [Formal Responder] → [Strict QA]
```

Фокус на структуре и ясности.

---

## Метрики качества контента

### KPI

| Метрика | Цель | Как измерить |
|---------|------|--------------|
| Читабельность | > 70% | Flesch Reading Ease |
| Грамматика | > 95% | Style Checker score |
| Фактическая точность | > 90% | QA validation |
| SEO оптимизация | > 80% | Keywords coverage |
| Уникальность | > 95% | Plagiarism check |

### Мониторинг через API

```python
# Получить статистику качества
stats = requests.get("http://localhost:8000/api/training/statistics").json()

print(f"Среднее качество: {stats['average_quality']:.1%}")
print(f"Успешность: {stats['success_rate']:.1%}")
```

---

## Масштабирование

### Batch обработка

```python
topics = [
    "AI в здравоохранении",
    "Blockchain технологии",
    "Квантовые вычисления",
    "IoT и умные города",
    "Кибербезопасность в 2026"
]

articles = []
for topic in topics:
    article = create_article_simple(topic)
    articles.append({
        "topic": topic,
        "content": article,
        "created_at": datetime.now()
    })
    
print(f"Создано статей: {len(articles)}")
```

### Параллельное выполнение

```python
import asyncio
import aiohttp

async def create_article_async(session, topic):
    async with session.post(
        "http://localhost:8000/api/chat/content/message/stream",
        data={"message": f"Создать статью: {topic}"}
    ) as response:
        return await response.text()

async def create_multiple_articles(topics):
    async with aiohttp.ClientSession() as session:
        tasks = [create_article_async(session, topic) for topic in topics]
        return await asyncio.gather(*tasks)

# Создать 10 статей параллельно
articles = asyncio.run(create_multiple_articles(topics))
```

---

## Итоги

### Вы создали

- Workflow для автоматического создания контента
- 5-шаговый процесс (research → outline → write → style → qa)
- Систему контроля качества
- Гибкую настройку под тип контента

### Вы узнали

- Как структурировать сложный creative workflow
- Как балансировать скорость и качество
- Как мониторить метрики
- Как масштабировать на batch обработку

### Применение

- Блог-посты
- Технические статьи
- Маркетинговые материалы
- Образовательный контент
- SEO тексты

---

**Следующий workflow**: [Анализ данных](data_analysis.md)
