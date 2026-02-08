# Туториал 4: Понимание инструментов

**Уровень**: Начинающий  
**Время**: 20 минут  
**Цель**: Понять как устроены инструменты

---

## Что вы узнаете

- Что такое инструмент (tool/agent)
- Как инструменты определяют свои возможности
- Что такое репутация и как она меняется
- Как инструменты связаны с LLM API
- Какие инструменты доступны

---

## Концепция инструмента

### Определение

**Инструмент** - это компонент, который:
- Принимает данные определенного формата (input connector)
- Выполняет одну задачу
- Возвращает данные определенного формата (output connector)

### Пример

```python
class GPT4Classifier:
    name = "GPT-4 Classifier"
    
    input = Connector("text", "question")   # Принимает текстовые вопросы
    output = Connector("text", "category")  # Возвращает категории
    
    reputation = 0.98  # Насколько хорошо работает (0-1)
    cost = 0.03        # Стоимость одного вызова ($)
    
    def execute(self, input_data):
        # Вызов GPT-4 API для классификации
        return classify_with_gpt4(input_data)
```

---

## Свойства инструмента

### Базовые метаданные

```python
tool.metadata:
    tool_name: str           # "GPT-4 Classifier"
    description: str         # Описание возможностей
    reputation: float        # 0.0-1.0, насколько хорошо работает
    mean_cost: float         # Средняя стоимость ($)
    mean_time_answer: float  # Среднее время ответа (секунды)
```

### Статистика для обучения

```python
tool.metadata:
    training_sample_size: int      # Сколько раз использовался
    variance_estimate: float       # Дисперсия качества
    quality_scores: List[float]    # История оценок
    capabilities_embedding: List[float]  # Векторное представление
```

### Коннекторы

```python
tool.input = Connector("text", "question")
# data_format="text", semantic_format="question"

tool.output = Connector("text", "category")
# data_format="text", semantic_format="category"
```

---

## Типы инструментов

### 1. Classifiers (Классификаторы)

**Назначение**: Категоризация текста

**Коннекторы**: `text|question` → `text|category`

**Примеры**:
- GPT-4 Classifier (высокая точность)
- Claude Classifier (понимание контекста)
- Local Classifier (быстрый, приватный)
- Fast Classifier (сверхбыстрый)

**Использование**:
- Классификация запросов поддержки
- Определение тональности
- Категоризация документов

### 2. Content Generators (Генераторы контента)

**Назначение**: Создание текста

**Коннекторы**: `text|outline` → `text|content`

**Примеры**:
- Creative Responder (креативные тексты)
- Formal Responder (официальный тон)
- Technical Writer (техническая документация)
- Friendly Responder (дружелюбный тон)

**Использование**:
- Написание статей
- Генерация ответов
- Создание документации

### 3. Quality Assurance (Контроль качества)

**Назначение**: Проверка и валидация

**Коннекторы**: `text|content` → `text|validated`

**Примеры**:
- Strict QA (строгая проверка)
- Balanced QA (сбалансированная)
- Fast QA (быстрая проверка)

**Использование**:
- Проверка сгенерированного контента
- Валидация ответов
- Финальный контроль

### 4. Analyzers (Анализаторы)

**Назначение**: Анализ и исследование

**Коннекторы**: `text|query` → `text|findings`

**Примеры**:
- Trend Analyzer (анализ трендов)
- Web Scraper (извлечение данных)

**Использование**:
- Исследование тем
- Анализ данных
- Поиск информации

### 5. Editors (Редакторы)

**Назначение**: Улучшение текста

**Коннекторы**: `text|draft` → `text|polished`

**Примеры**:
- Style Checker (проверка стиля)
- Style Improver (улучшение стиля)

**Использование**:
- Редактирование текстов
- Проверка грамматики
- Улучшение формулировок

---

## Как инструменты выбираются

### Входные данные для выбора

```python
Задача: "Классифицировать отзыв клиента"
    ↓
1. NLI определяет коннекторы:
   text|question → text|category
   
2. GraphStrategyFinder ищет инструменты с этими коннекторами:
   - GPT-4 Classifier ✓
   - Claude Classifier ✓
   - Local Classifier ✓
   - QA tools ✗ (другие коннекторы)
   
3. InstrumentSelector выбирает из кандидатов:
   Логиты → Softmax → Вероятностный выбор
```

### Факторы выбора

1. **Сходство с задачей**:
   ```
   cos_sim(task_embedding, tool_embedding)
   ```

2. **Репутация инструмента**:
   ```
   log(reputation)  # Выше репутация → выше шанс
   ```

3. **История использования**:
   ```
   Температура снижается с опытом
   ```

---

## Жизненный цикл инструмента

### 1. Создание (в БД)

```sql
INSERT INTO agents (id, name, type, reputation, ...)
VALUES ('agent-new', 'My Tool', 'custom', 0.5, ...);
```

Начальная репутация: 0.5 (средняя неопределенность)

### 2. Начало использования

```
Выполнение 1:
  Выбран с вероятностью 0.25 (среди 4 кандидатов)
  Качество: 0.82
  Репутация: 0.5 → 0.532 (+0.032)

Выполнение 2:
  Выбран с вероятностью 0.29 (репутация выросла)
  Качество: 0.91
  Репутация: 0.532 → 0.568 (+0.036)
```

### 3. После обучения (100 использований)

```
Репутация: 0.5 → 0.87 (+0.37)
Sample size: 0 → 100
Variance: 0.2 → 0.05 (меньше неопределенности)
Температура: высокая → низкая
Вероятность выбора: 25% → 65%
```

### 4. Стабилизация

После достаточного количества данных:
- Репутация стабилизируется
- Температура минимальная
- Выбор почти детерминированный

---

## Просмотр информации об инструментах

### Через API

```bash
# Все инструменты
curl http://localhost:8000/api/agents-library
```

Ожидаемый вывод:
```json
{
  "agents": [
    {
      "id": "agent-accurate-qa",
      "name": "Accurate QA",
      "icon": "Q6",
      "color": "#ef4444",
      "type": "qa",
      "specialization": "High accuracy QA",
      "capabilities": [
        "accuracy"
      ],
      "metrics": {
        "avgResponseTime": 4000,
        "avgScore": 0.92
      }
    }
  ],
  ...
}
```

Попробуйте самостоятельно:
```bash
# Метрики конкретного
curl http://localhost:8000/api/training/tools/agent-classifier-gpt4
```

### Через БД

```python
import sqlite3

conn = sqlite3.connect('grapharchitect.db')
cursor = conn.cursor()

# Все инструменты
cursor.execute("SELECT name, type, cost FROM agents ORDER BY name")
for row in cursor.fetchall():
    print(f"{row[0]:30} {row[1]:20} ${row[2]:.3f}")

# Метрики обучения
cursor.execute("""
    SELECT agent_id, reputation, training_sample_size
    FROM tool_metrics
    ORDER BY reputation DESC
""")
```

---

## Добавление своего инструмента

### Через db_manager

```bash
python db_manager.py add_agent

# Интерактивный ввод:
ID: agent-my-tool
Name: My Custom Tool
Type: custom
Icon: CT
Color: #ff6b6b
Specialization: My specific task
Cost: 0.01
```

### Через код

```python
from models import Agent
from repository import get_repository

# Создаем агента
agent = Agent(
    id="agent-my-analyzer",
    name="My Data Analyzer",
    type="analysis",
    icon="MA",
    color="#4ecdc4",
    specialization="Specific data analysis",
    capabilities=["data_analysis", "custom_metrics"],
    cost=0.015,
    metrics={
        "avgResponseTime": 2000,
        "avgScore": 0.8
    }
)

# Сохраняем в БД
repo = get_repository()
repo.save_agent(agent)

print(f"Agent added: {agent.name}")
```

### Реализация execute()

Для реального использования нужна реализация:

```python
# В grapharchitect/tools/ создайте свой инструмент
class MyAnalyzerTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.metadata.tool_name = "My Data Analyzer"
        self.input = Connector("text", "data")
        self.output = Connector("text", "analysis")
    
    def execute(self, input_data):
        # Ваша логика
        import pandas as pd
        
        # Парсинг данных
        df = pd.read_csv(StringIO(input_data))
        
        # Анализ
        summary = df.describe().to_string()
        
        return f"Analysis results:\n{summary}"
```

---

## Управление инструментами

### Включение/выключение

```python
# Временно отключить инструмент
repo = get_repository()
agent = repo.get_agent("agent-expensive-tool")
agent.metrics['enabled'] = False
repo.save_agent(agent)
```

### Изменение приоритета

```python
# Увеличить репутацию вручную
agent = repo.get_agent("agent-my-tool")
agent.metrics['avgScore'] = 0.95  # Высокая репутация
repo.save_agent(agent)
```

---

## Упражнения

### Упражнение 1: Изучите инструменты

```bash
python db_manager.py list_agents
```

**Задание**: Составьте таблицу:
- Тип инструмента
- Входной/выходной коннектор
- Стоимость
- Репутация

### Упражнение 2: Добавьте инструмент

Добавьте свой инструмент для specific задачи.

**Вопрос**: Как часто он выбирается изначально?

### Упражнение 3: Отследите обучение

Выполните задачу 10 раз и смотрите как меняется репутация:

```bash
curl http://localhost:8000/api/training/tools/agent-my-tool
```

---

## Итоги

### Вы узнали

- Инструменты определяются коннекторами
- Репутация влияет на выбор
- Метаданные используются для обучения
- Инструменты можно добавлять динамически
- Система адаптируется к новым инструментам

### Ключевые концепции

- Tool = специализированная задача
- Connectors = интерфейс инструмента
- Reputation = качество работы
- Training = улучшение со временем

---

**Следующий**: [Простое использование API](05_simple_api_usage.md)
