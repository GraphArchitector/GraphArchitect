# Workflow: Обработка документов

**Сценарий**: Автоматическая обработка и анализ документов  
**Сложность**: Средняя  
**Время**: 20 минут

---

## Описание задачи

Обработка документов включает:
1. Извлечение текста из различных форматов
2. Структурирование и очистка
3. Анализ содержимого
4. Извлечение ключевой информации
5. Создание summary или отчета

---

## Архитектура Workflow

```
Документ (PDF/Word/HTML)
    ↓
┌─────────────────────────────────┐
│ Шаг 1: Text Extraction          │
│ - Извлечение текста             │
│ - OCR для изображений           │
│ - Сохранение структуры          │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 2: Content Analysis         │
│ - Определение типа документа    │
│ - Извлечение метаданных         │
│ - Определение ключевых тем      │
│ Инструменты:                    │
│   - Fast Parser                 │
│   - Classifier                  │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 3: Information Extraction   │
│ - Извлечение сущностей (NER)    │
│ - Извлечение дат, чисел         │
│ - Извлечение ключевых фактов    │
│ Инструменты:                    │
│   - Technical Writer            │
│   - Trend Analyzer              │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 4: Summarization            │
│ - Краткое резюме                │
│ - Ключевые пункты               │
│ - Actionable items              │
│ Инструменты:                    │
│   - Summary Reporter            │
└────────────┬────────────────────┘
             ↓
Обработанный документ + Summary
```

---

## Реализация

### Обработка одного документа

```python
import requests

def process_document(file_path, document_type="general"):
    """
    Обработать документ через GraphArchitect.
    
    Args:
        file_path: Путь к документу
        document_type: Тип документа (contract, report, article)
        
    Returns:
        Результаты обработки
    """
    # Шаг 1: Загрузка документа
    with open(file_path, 'rb') as f:
        upload_response = requests.post(
            "http://localhost:8000/api/chat/docprocessing/document",
            files={'file': f}
        )
    
    doc_id = upload_response.json()['document_id']
    
    # Шаг 2: Обработка
    message = f"""
    Обработать документ {doc_id}:
    
    Тип документа: {document_type}
    
    Требуется:
    1. Извлечь ключевую информацию
    2. Создать структурированное резюме
    3. Выявить важные даты, суммы, имена
    4. Сформулировать actionable items
    """
    
    response = requests.post(
        "http://localhost:8000/api/chat/docprocessing/message/stream",
        data={
            "message": message,
            "planning_algorithm": "yen_5"
        },
        stream=True
    )
    
    result = {
        "document_id": doc_id,
        "summary": "",
        "key_info": [],
        "action_items": []
    }
    
    for line in response.iter_lines():
        if line:
            import json
            chunk = json.loads(line)
            
            if chunk['type'] == 'text':
                result['summary'] += chunk['content']
    
    return result


# Использование
result = process_document(
    "contract.pdf",
    document_type="contract"
)

print("Summary:")
print(result['summary'])
```

---

## Типы документов

### Контракты и юридические документы

```python
message = """
Проанализировать контракт:

Извлечь:
- Стороны договора
- Ключевые условия
- Даты и дедлайны
- Финансовые обязательства
- Риски и ограничения
"""
```

**Workflow**: Parser → Technical Writer → Strict QA

### Отчеты и презентации

```python
message = """
Обработать отчет:

Создать:
- Executive summary
- Ключевые метрики
- Основные выводы
- Рекомендации
"""
```

**Workflow**: Parser → Trend Analyzer → Summary Reporter

### Технические документы

```python
message = """
Проанализировать техническую документацию:

Извлечь:
- API endpoints
- Параметры и типы данных
- Примеры использования
- Ограничения
"""
```

**Workflow**: Parser → Technical Writer → Style Checker

---

## Batch обработка

### Обработка папки документов

```python
import os
from pathlib import Path

def process_folder(folder_path):
    """Обработать все документы в папке"""
    
    results = []
    
    for file_path in Path(folder_path).glob("*.pdf"):
        print(f"Processing: {file_path.name}")
        
        result = process_document(str(file_path))
        results.append({
            "filename": file_path.name,
            "result": result
        })
    
    return results


# Использование
results = process_folder("./documents/contracts/")

# Создание сводного отчета
summary = create_summary_report(results)
```

### Мониторинг папки

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Новый документ появился"""
        if event.src_path.endswith('.pdf'):
            print(f"New document: {event.src_path}")
            
            # Автоматическая обработка
            result = process_document(event.src_path)
            
            # Сохранение результата
            save_processing_result(result)

# Запуск мониторинга
observer = Observer()
observer.schedule(DocumentHandler(), path='./inbox/', recursive=False)
observer.start()
```

---

## Извлечение структурированных данных

### Из накладных и счетов

```python
message = """
Извлечь из счета:

Структура:
- Номер счета
- Дата
- Поставщик (название, адрес, ИНН)
- Получатель (название, адрес, ИНН)
- Позиции (название, количество, цена)
- Итоговая сумма
- НДС

Вернуть в формате JSON
"""
```

**Результат**: Структурированные данные для импорта в систему

### Из резюме кандидатов

```python
message = """
Извлечь из резюме:

Данные кандидата:
- ФИО
- Контакты
- Опыт работы (компания, должность, период)
- Образование
- Навыки
- Языки

Также оценить:
- Соответствие позиции: {job_description}
- Сильные стороны
- Потенциальные риски
"""
```

---

## Метрики обработки

### KPI

| Метрика | Цель | Измерение |
|---------|------|-----------|
| Точность извлечения | > 95% | Сравнение с эталоном |
| Полнота данных | > 90% | % извлеченных полей |
| Скорость обработки | < 10s/doc | Время на документ |
| Стоимость | < $0.05/doc | API costs |

### Мониторинг

```python
# Статистика обработки
stats = requests.get("http://localhost:8000/api/training/statistics").json()

print(f"Обработано документов: {stats['total_executions']}")
print(f"Средняя точность: {stats['average_quality']:.1%}")
print(f"Среднее время: {stats['average_execution_time']:.1f}s")
```

---

## Интеграция с DMS

### Автоматическая категоризация

```python
def auto_categorize_document(file_path):
    """Автоматическое определение категории документа"""
    
    # Обработка
    result = process_document(file_path)
    
    # Определение категории
    category_response = requests.post(
        "http://localhost:8000/api/chat/classify/message",
        data={
            "message": f"Определить категорию документа: {result['summary']}"
        }
    )
    
    category = category_response.json()['category']
    
    # Перемещение в соответствующую папку
    import shutil
    dest = f"./archive/{category}/{Path(file_path).name}"
    shutil.move(file_path, dest)
    
    return category
```

---

## Итоги

### Вы создали

- Workflow для автоматической обработки документов
- 4-шаговый процесс (extraction → analysis → extraction → summary)
- Batch обработку
- Интеграцию с DMS

### Применение

- Обработка контрактов
- Анализ отчетов
- Извлечение данных из счетов
- Обработка резюме
- Архивирование документов

---

**Все workflows**: [INDEX](../INDEX.md)
