# Workflow: Анализ данных

**Сценарий**: Автоматический анализ данных и создание отчетов  
**Сложность**: Средняя  
**Время**: 25 минут

---

## Описание задачи

Автоматический анализ данных включает:
1. Извлечение и парсинг данных
2. Статистический анализ
3. Выявление трендов и паттернов
4. Создание визуализаций
5. Генерация отчета

---

## Архитектура Workflow

```
Сырые данные
    ↓
┌─────────────────────────────────┐
│ Шаг 1: Data Parsing             │
│ - Извлечение данных             │
│ - Очистка и нормализация        │
│ - Валидация                     │
│ Инструменты:                    │
│   - Fast Parser                 │
│   - Web Scraper                 │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 2: Analysis                 │
│ - Статистические метрики        │
│ - Корреляции                    │
│ - Аномалии                      │
│ Инструменты:                    │
│   - Trend Analyzer              │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 3: Insights Generation      │
│ - Выявление паттернов           │
│ - Бизнес-инсайты                │
│ - Рекомендации                  │
│ Инструменты:                    │
│   - GPT-4 Classifier            │
│   - Claude Classifier           │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 4: Report Generation        │
│ - Структурирование              │
│ - Форматирование                │
│ - Добавление выводов            │
│ Инструменты:                    │
│   - Summary Reporter            │
│   - Technical Writer            │
└────────────┬────────────────────┘
             ↓
Аналитический отчет
```

---

## Реализация

### Пример 1: Анализ продаж

```python
import requests
import json

def analyze_sales_data(data_description):
    """
    Анализ данных о продажах.
    
    Args:
        data_description: Описание данных для анализа
        
    Returns:
        Аналитический отчет
    """
    message = f"""
    Проанализировать данные о продажах:
    {data_description}
    
    Необходимо:
    1. Выявить тренды продаж по месяцам
    2. Определить топ-продукты
    3. Найти аномалии и выбросы
    4. Предложить рекомендации для роста
    """
    
    response = requests.post(
        "http://localhost:8000/api/chat/analysis/message/stream",
        data={
            "message": message,
            "planning_algorithm": "yen_5"
        },
        stream=True
    )
    
    report = ""
    steps_info = []
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            
            if chunk['type'] == 'agent_selected':
                steps_info.append({
                    'step': chunk.get('step_id'),
                    'tool': chunk['agent_id'],
                    'probability': chunk['score']
                })
            
            elif chunk['type'] == 'text':
                report += chunk['content']
    
    return {
        'report': report,
        'workflow_steps': steps_info
    }


# Использование
result = analyze_sales_data("""
Данные за Q1 2026:
- Январь: 150 продаж, $45,000
- Февраль: 180 продаж, $52,000
- Март: 220 продаж, $68,000

Топ-3 продукта:
1. Product A: 120 единиц
2. Product B: 95 единиц
3. Product C: 75 единиц
""")

print("Отчет:")
print(result['report'])

print("\nИспользованные инструменты:")
for step in result['workflow_steps']:
    print(f"  {step['step']}: {step['tool']} (p={step['probability']:.3f})")
```

---

## Типовые аналитические задачи

### 1. Анализ временных рядов

```python
message = """
Проанализировать временной ряд:
- Выявить тренд (рост/падение)
- Найти сезонность
- Предсказать следующий период
"""
```

**Инструменты**:
- Trend Analyzer → выявление трендов
- Technical Writer → отчет

### 2. Сегментация клиентов

```python
message = """
Сегментировать клиентов на основе:
- Частоты покупок
- Среднего чека
- Категорий товаров
"""
```

**Инструменты**:
- Classifier → сегментация
- Summary Reporter → описание сегментов

### 3. Анализ текстовых отзывов

```python
message = """
Проанализировать отзывы клиентов:
- Определить общий sentiment
- Выявить основные темы
- Найти проблемные точки
"""
```

**Инструменты**:
- Classifier → sentiment analysis
- Trend Analyzer → темы и паттерны
- Summary Reporter → итоговый отчет

---

## Интеграция с данными

### Загрузка CSV/Excel

```python
# Загрузить файл данных
files = {'file': open('sales_data.csv', 'rb')}
response = requests.post(
    "http://localhost:8000/api/chat/analysis/document",
    files=files
)

document_id = response.json()['document_id']

# Анализ загруженных данных
response = requests.post(
    "http://localhost:8000/api/chat/analysis/message/stream",
    data={
        "message": f"Проанализировать загруженный файл {document_id}",
        "planning_algorithm": "yen_5"
    },
    stream=True
)
```

### Подключение к БД

```python
# В собственном инструменте
class DatabaseAnalyzerTool(BaseTool):
    def execute(self, input_data):
        # Подключение к вашей БД
        import psycopg2
        conn = psycopg2.connect(...)
        
        # SQL запрос
        df = pd.read_sql("SELECT * FROM sales", conn)
        
        # Анализ
        summary = df.describe()
        
        return f"Статистика: {summary}"
```

---

## Визуализация результатов

### Создание графиков

Пока инструменты возвращают текст, но можно расширить:

```python
class VisualizationTool(BaseTool):
    def __init__(self):
        self.input = Connector("text", "data")
        self.output = Connector("image", "chart")
    
    def execute(self, input_data):
        import matplotlib.pyplot as plt
        
        # Парсинг данных
        data = parse_data(input_data)
        
        # Создание графика
        plt.figure()
        plt.plot(data)
        plt.savefig("chart.png")
        
        return "chart.png"
```

---

## Метрики аналитических workflow

### KPI

| Метрика | Цель | Измерение |
|---------|------|-----------|
| Точность анализа | > 90% | Сравнение с экспертом |
| Полнота инсайтов | > 85% | % выявленных паттернов |
| Скорость | < 30s | Время выполнения |
| Стоимость | < $0.20 | API costs |
| Actionability | > 80% | % применимых рекомендаций |

---

## Итоги

### Вы создали

- Workflow для автоматического анализа данных
- 4-шаговый процесс (парсинг → анализ → инсайты → отчет)
- Интеграцию с внешними данными
- Систему метрик качества

### Применение

- Анализ продаж
- Анализ отзывов
- Финансовый анализ
- Маркетинговая аналитика
- Исследование трендов

---

**Следующий workflow**: [Code Review](code_review.md)
