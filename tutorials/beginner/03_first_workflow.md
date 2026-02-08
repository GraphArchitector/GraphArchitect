# Туториал 3: Создание первого Workflow

**Уровень**: Начинающий  
**Время**: 20 минут  
**Цель**: Создать и выполнить собственный workflow

---

## Что вы узнаете

- Как создать простой workflow
- Как определить шаги workflow
- Как указать кандидатов для каждого шага
- Как выполнить workflow через API
- Как интерпретировать результаты

---

## Сценарий: Обработка запроса клиента

Создадим workflow для обработки запросов службы поддержки:

```
Запрос клиента
    ↓
[Шаг 1] Классификация запроса
    ↓
[Шаг 2] Генерация ответа
    ↓
[Шаг 3] Проверка качества
    ↓
Готовый ответ
```

---

## Шаг 1: Понимание структуры Workflow

### WorkflowChain

```python
class WorkflowChain:
    chat_id: str              # ID чата
    name: str                 # Название workflow
    description: str          # Описание
    steps: List[WorkflowStep] # Шаги выполнения
    agents: List[str]         # ID используемых агентов
```

### WorkflowStep

```python
class WorkflowStep:
    id: str                      # ID шага
    name: str                    # Название (напр., "Classification")
    order: int                   # Порядковый номер
    candidate_agents: List[str]  # ID кандидатов для выбора
    selection_criteria: SelectionCriteria  # Критерии выбора
```

---

## Шаг 2: Определение шагов

### Шаг 1: Классификация

**Цель**: Определить тип запроса (вопрос/жалоба/предложение)

**Кандидаты**:
- GPT-4 Classifier (точный, дорогой)
- Claude Classifier (точный, средний)
- Local Classifier (быстрый, дешевый)

**Критерий выбора**: best_quality_score

### Шаг 2: Генерация ответа

**Цель**: Создать ответ на основе типа запроса

**Кандидаты**:
- Formal Responder (официальный тон)
- Friendly Responder (дружелюбный тон)
- Technical Responder (технический ответ)

**Критерий**: balanced

### Шаг 3: Проверка качества

**Цель**: Валидировать ответ

**Кандидаты**:
- Strict QA (строгая проверка)
- Balanced QA (сбалансированная)

**Критерий**: best_quality_score

---

## Шаг 3: Создание через API

### Вариант 1: Использовать готовый шаблон

```bash
curl -X POST http://localhost:8000/api/chat/demo/message/stream \
  -F "message=Клиент недоволен задержкой доставки" \
  -F "planning_algorithm=yen_5"
```

GraphArchitect автоматически:
1. Распарсит запрос через NLI
2. Найдет оптимальную стратегию
3. Выберет инструменты через softmax
4. Выполнит и обучится

### Вариант 2: Через Python код

```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat/demo/message/stream",
    data={
        "message": "Клиент недоволен задержкой доставки",
        "planning_algorithm": "yen_5"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        import json
        chunk = json.loads(line)
        
        if chunk['type'] == 'agent_selected':
            print(f"Выбран: {chunk['agent_id']}")
            print(f"Вероятность: {chunk['score']:.3f}")
```

---

## Шаг 4: Выполнение и наблюдение

### Отправьте запрос

Через Web интерфейс или API отправьте:

```
Клиент жалуется на медленную доставку и хочет возврат средств
```

### Наблюдайте процесс

**В Web интерфейсе вы увидите**:

```
[Генерация графа]
  Фаза 1: Поиск архитектур в k-NN...
  Фаза 2: Генерация 5 вариантов (Yen)
  Фаза 3: LLM-синтез из Top-5 путей
  
[Шаг 1: Classify Request]
  Кандидаты:
    - GPT-4 Classifier (прогресс: 100%, score: 0.87)
    - Claude Classifier (прогресс: 100%, score: 0.65)
    - Local Classifier (прогресс: 100%, score: 0.23)
  Выбран: GPT-4 Classifier (вероятность: 0.654)
  
[Выполнение]
  [GPT-4 Classifier] Analyzing context...
  [GPT-4 Classifier] Generating solution...
  [GPT-4 Classifier] Validating result...
  
[Шаг 2: Generate Response]
  ...
```

---

## Шаг 5: Анализ результатов

### Проверьте метаданные

В логах сервера:

```
INFO - Executing task: Клиент жалуется...
INFO - Found 3 strategies
INFO - Selected: GPT-4 Classifier
INFO - Steps: 3, Time: 4.52s, Cost: 0.078
INFO - Task completed: COMPLETED
INFO - Trained tools: 3
```

### Проверьте обучение

```bash
curl http://localhost:8000/api/training/statistics
```

Результат:
```json
{
  "enabled": true,
  "total_executions": 1,
  "average_quality": 0.870,
  "success_rate": 1.000
}
```

---

## Упражнения

### Упражнение 1: Разные алгоритмы

Отправьте тот же запрос с разными алгоритмами:

```bash
# Dijkstra (один путь)
planning_algorithm=dijkstra

# Yen топ-3 (три пути)
planning_algorithm=yen_3

# ACO (муравьиный)
planning_algorithm=ant_5
```

**Вопрос**: Какой алгоритм нашел больше стратегий?

### Упражнение 2: Разные запросы

Попробуйте спросить:

```
Что такое AI?
```

```
Создай план статьи о Python
```

```
Суммировать этот длинный текст: "..."
```

**Вопрос**: Какие инструменты выбрались для каждого?

### Упражнение 3: Обратная связь

```bash
curl -X POST http://localhost:8000/api/training/feedback \
  -F "task_id=<uuid-from-execution>" \
  -F "quality_score=0.95"
```

**Вопрос**: Как изменилась репутация инструмента?

---

## Типичные ошибки

### Ошибка 1: "No strategies found"

**Причина**: Коннекторы не совпадают

**Решение**: 
- Проверьте доступные инструменты
- Убедитесь что есть путь от входа к выходу

### Ошибка 2: "All scores are equal"

**Причина**: Слишком высокая температура

**Решение**: Подождите, после обучения температура снизится

### Ошибка 3: "Execution failed"

**Причина**: Ошибка в инструменте

**Решение**: Проверьте логи, система использует fallback

---

## Итоги

### Вы научились

- Понимать структуру workflow
- Определять шаги и кандидатов
- Выполнять workflow через API
- Наблюдать процесс выполнения
- Анализировать результаты

### Ключевые моменты

- Workflow = последовательность шагов
- Каждый шаг = выбор из кандидатов
- Выбор = softmax вероятностный
- Результат = обучение системы

---

**Время выполнения**: 20 минут  
**Практика**: 3 упражнения  
**Следующий**: [Understanding Tools](04_understanding_tools.md)
