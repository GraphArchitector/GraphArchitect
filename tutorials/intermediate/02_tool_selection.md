# Туториал: Система выбора инструментов

**Уровень**: Средний  
**Время**: 30 минут  
**Цель**: Понять как GraphArchitect выбирает инструменты

---

## Что вы узнаете

- Как работает InstrumentSelector
- Что такое логиты и как они вычисляются
- Как работает адаптивная температура
- Что такое softmax и зачем он нужен
- Как происходит вероятностное сэмплирование

---

## Проблема выбора

### Сценарий

Есть задача "Классифицировать текст" и 4 кандидата:

| Инструмент | Репутация | Стоимость | Скорость |
|------------|-----------|-----------|----------|
| GPT-4 Classifier | 0.98 | $0.03 | 2.8s |
| Claude Classifier | 0.95 | $0.02 | 3.2s |
| Local Classifier | 0.78 | $0.001 | 1.2s |
| Fast Classifier | 0.72 | $0.005 | 0.8s |

**Вопрос**: Какой выбрать?

### Наивные подходы

**1. Всегда лучший (эксплуатация)**:
```python
selected = max(tools, key=lambda t: t.reputation)
# Всегда GPT-4, но дорого
```

**2. Случайный (исследование)**:
```python
selected = random.choice(tools)
# Изучаем все, но неэффективно
```

**3. Round-robin (справедливость)**:
```python
selected = tools[counter % len(tools)]
# Все поровну, но игнорирует качество
```

### Проблема

Нужен **баланс** между:
- Эксплуатацией (использовать лучшие)
- Исследованием (пробовать новые)
- Обучением (собирать данные)

---

## Решение: Softmax с адаптивной температурой

### Алгоритм (5 шагов)

#### Шаг 1: Вычисление логитов

```python
logit = косинусное_сходство(задача, инструмент) + log(reputation)
```

**Для нашего примера**:

| Инструмент | Cos_sim | log(rep) | Логит |
|------------|---------|----------|-------|
| GPT-4 | 0.92 | -0.02 | 0.90 |
| Claude | 0.87 | -0.05 | 0.82 |
| Local | 0.65 | -0.25 | 0.40 |
| Fast | 0.58 | -0.33 | 0.25 |

**Интуиция**: 
- Высокое сходство + высокая репутация = высокий логит
- Логит отражает "подходимость" инструмента

#### Шаг 2: Отбор топ-K

```python
top_k = sorted(tools, key=lambda t: t.logit, reverse=True)[:K]
```

Берем K=3 лучших:
- GPT-4 (0.90)
- Claude (0.82)
- Local (0.40)

Fast отсеивается (слишком низкий логит).

#### Шаг 3: Вычисление температуры

```python
T_group = (C/K) * Σ √(variance_k / sample_size_k)
```

Для каждого инструмента:
```
T_GPT4 = √(0.01 / 150) = 0.008  # Много данных, малая дисперсия
T_Claude = √(0.05 / 80) = 0.025
T_Local = √(0.15 / 30) = 0.071  # Мало данных, большая дисперсия
```

Температура группы:
```
T_group = (1.0/3) * (0.008 + 0.025 + 0.071) = 0.035
```

**Интуиция**:
- Низкая T → концентрация на лучших (эксплуатация)
- Высокая T → равномерное распределение (исследование)
- T снижается по мере обучения

#### Шаг 4: Softmax

```python
P(k) = exp(logit_k / T) / Σ exp(logit_i / T)
```

С температурой T=0.035:

```
exp(0.90/0.035) = exp(25.7) = огромное число
exp(0.82/0.035) = exp(23.4) = большое число
exp(0.40/0.035) = exp(11.4) = среднее число

P(GPT-4) = огромное / (огромное + большое + среднее) ≈ 0.73
P(Claude) = большое / ... ≈ 0.25
P(Local) = среднее / ... ≈ 0.02
```

**Интуиция**: 
- Softmax превращает логиты в вероятности
- Сумма вероятностей = 1.0
- Лучшие имеют выше шанс, но не гарантировано

#### Шаг 5: Сэмплирование

```python
# Рулетка по вероятностям
random_value = random.random()  # 0.0-1.0

if random_value < 0.73:
    selected = GPT-4
elif random_value < 0.73 + 0.25:
    selected = Claude
else:
    selected = Local
```

**Результат**: 
- GPT-4: 73% шанс
- Claude: 25% шанс
- Local: 2% шанс

---

## Влияние температуры

### Низкая температура (T=0.01)

```
Логиты: [0.90, 0.82, 0.40]
Softmax(T=0.01):
  P(GPT-4) = 0.95  # Почти всегда
  P(Claude) = 0.05
  P(Local) = 0.00
```

**Эффект**: Почти детерминированный выбор лучшего.

### Средняя температура (T=0.5)

```
Softmax(T=0.5):
  P(GPT-4) = 0.65
  P(Claude) = 0.30
  P(Local) = 0.05
```

**Эффект**: Баланс эксплуатации и исследования.

### Высокая температура (T=5.0)

```
Softmax(T=5.0):
  P(GPT-4) = 0.40
  P(Claude) = 0.35
  P(Local) = 0.25
```

**Эффект**: Почти равномерное распределение.

---

## Адаптивная температура

### Как изменяется со временем

```
Начало (10 примеров):
  variance = 0.15 (большая неопределенность)
  sample_size = 10
  T = √(0.15/10) = 0.122  # Высокая

После обучения (100 примеров):
  variance = 0.05 (меньше неопределенности)
  sample_size = 100
  T = √(0.05/100) = 0.022  # Низкая
```

**Эффект**: 
- Вначале много исследования
- Постепенно больше эксплуатации
- Система становится увереннее

---

## Практический пример

### Код для тестирования

```python
# В examples/Python создайте:
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "GraphArchitectLib"))

from grapharchitect.services.selection.instrument_selector import InstrumentSelector
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector


class MockTool(BaseTool):
    def __init__(self, name, reputation):
        super().__init__()
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
        self.metadata.training_sample_size = 50
        self.metadata.variance_estimate = 0.1
        
        self.input = Connector("text", "question")
        self.output = Connector("text", "category")
    
    def execute(self, input_data):
        return f"[{self.metadata.tool_name}] Processed"


# Создаем инструменты
tools = [
    MockTool("GPT-4", 0.98),
    MockTool("Claude", 0.95),
    MockTool("Local", 0.78),
    MockTool("Fast", 0.72)
]

# Embedding service
embedding = SimpleEmbeddingService(dimension=384)

# Создаем эмбеддинги
task_embedding = embedding.embed_text("Классифицировать текст")
for tool in tools:
    tool.metadata.capabilities_embedding = embedding.embed_tool_capabilities(tool)

# Selector
selector = InstrumentSelector(temperature_constant=1.0)

# Выбор (повторяем 100 раз)
selections = {}
for _ in range(100):
    result = selector.select_instrument(tools, task_embedding, top_k=3)
    name = result.selected_tool.metadata.tool_name
    selections[name] = selections.get(name, 0) + 1

# Результаты
print("Распределение выборов (100 итераций):")
for name, count in sorted(selections.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {count}% (теоретически: {count/100:.1%})")
```

**Ожидаемый вывод**:

```
Распределение выборов (100 итераций):
  GPT-4: 73% (теоретически: 73.0%)
  Claude: 25% (теоретически: 25.0%)
  Local: 2% (теоретически: 2.0%)
  Fast: 0% (не в топ-3)
```

---

## Влияние на обучение

### Gradient Trace

При каждом выборе сохраняется:

```python
GradientTrace:
  task_embedding: [0.1, 0.2, ...]
  candidate_tools: [GPT-4, Claude, Local]
  logits: {GPT-4: 0.90, Claude: 0.82, Local: 0.40}
  probabilities: {GPT-4: 0.73, Claude: 0.25, Local: 0.02}
  selected_tool: GPT-4
  temperature: 0.035
```

Эта информация используется для обучения через Policy Gradient.

---

## Настройка параметров

### Константа температуры

```python
# В config.py
TEMPERATURE_CONSTANT = 1.0  # Стандарт

# Для большей эксплуатации
TEMPERATURE_CONSTANT = 0.1  # Почти всегда лучший

# Для большего исследования
TEMPERATURE_CONSTANT = 10.0  # Равномерное распределение
```

### Топ-K

```python
# Берем только топ-3
selector.select_instrument(tools, embedding, top_k=3)

# Берем всех
selector.select_instrument(tools, embedding, top_k=len(tools))
```

---

## Упражнения

### Упражнение 1: Эксперимент с температурой

Измените в config.py:
```python
TEMPERATURE_CONSTANT = 0.1  # Низкая
TEMPERATURE_CONSTANT = 1.0  # Средняя
TEMPERATURE_CONSTANT = 10.0  # Высокая
```

Выполните один и тот же запрос 10 раз для каждой температуры.

**Вопрос**: Как часто выбирался GPT-4 при разных T?

### Упражнение 2: Анализ логитов

Добавьте logging в grapharchitect_bridge.py:

```python
logger.debug(f"Logits: {selection_result.logits}")
logger.debug(f"Probabilities: {selection_result.probabilities}")
logger.debug(f"Temperature: {selection_result.temperature}")
```

**Вопрос**: Как соотносятся логиты и вероятности?

---

## Итоги

### Вы узнали

- Логиты = сходство + репутация
- Температура = неопределенность
- Softmax = логиты → вероятности
- Сэмплирование = вероятностный выбор
- Адаптивность = температура меняется

### Математика

Формулы понятны и применимы:
```
logit_i = cos_sim(task, tool_i) + log(reputation_i)
T = (C/K) * Σ√(D_i / m_i)
P(i) = exp(logit_i/T) / Σ exp(logit_j/T)
```

### Следующий шаг

[Система обучения](04_training_system.md) - как улучшаются инструменты

---

**Время выполнения**: 30 минут  
**Сложность**: Средняя  
**Требует**: Понимание вероятностей
