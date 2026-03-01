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
        "conclusions": "",
        "raw_text": ""
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
                
                sections["raw_text"] += content
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

print("\nRaw Text:")
print(research['raw_text'])
```

---

## Типы исследований

Следующие примеры требуют самостоятельной работы. 

### 1. Обзор литературы (Literature Review)

```python
message = f"""
Провести обзор литературы по теме: {topic}

Требования:
- Найти ключевые публикации (последние 3 года)
- Выявить основные направления исследований
- Определить пробелы в знаниях
- Предложить направления для будущих исследований
"""
```

### 2. Сравнительный анализ (Comparative Analysis)

```python
message = f"""
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

### 3. Анализ трендов (Trend Analysis)

```python
message = f"""
Проанализировать тренды в {domain} за последние {period}:

- Основные тренды
- Темпы роста
- Драйверы изменений
- Прогноз на будущее
"""
```

---

## Примеры исследовательских вопросов

### Технические исследования

```
"Сравнение архитектур нейронных сетей для NLP задач"
```

### Бизнес-исследования

```
"Анализ рынка электромобилей в 2026 году"
```

### Академические исследования

```
"Этические вопросы использования AI в медицине"
```

---

## Итоги

### Вы создали

- Workflow для автоматизации исследований
- 4-шаговый процесс (сбор → анализ → синтез → отчет)
- Систему контроля качества

### Применение

- Обзоры литературы
- Сравнительный анализ
- Анализ трендов
- Due diligence
- Конкурентный анализ