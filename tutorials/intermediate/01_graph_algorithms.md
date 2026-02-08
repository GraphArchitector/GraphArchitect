# Туториал: Алгоритмы поиска в графе

**Уровень**: Средний  
**Время**: 30 минут  
**Цель**: Понять алгоритмы поиска путей в графе инструментов

---

## Что вы узнаете

- Как работает граф инструментов
- Dijkstra - кратчайший путь
- A* - поиск с эвристикой
- Yen - топ-K путей
- Ant Colony - вероятностный поиск
- Когда какой алгоритм использовать

---

## Граф инструментов

### Структура

```
Вершины = Коннекторы (форматы данных)
Ребра = Инструменты (преобразования)

Пример:
  [text|question] ─[Classifier]→ [text|category]
  [text|question] ─[QA System]→ [text|answer]
  [text|outline] ─[Writer]→ [text|content]
```

### Веса ребер

```python
weight = -log(reputation)
```

**Примеры**:
- Reputation 0.98 → weight = -log(0.98) = 0.02
- Reputation 0.50 → weight = -log(0.50) = 0.69
- Reputation 0.10 → weight = -log(0.10) = 2.30

**Логика**: Чем выше репутация, тем меньше вес (лучше путь)

---

## Алгоритм 1: Dijkstra

### Назначение

Найти **один кратчайший путь** от входа к выходу.

### Как работает

```
1. Инициализация:
   distance[start] = 0
   distance[other] = ∞

2. Priority queue с вершинами по distance

3. Для каждой вершины v:
   Для каждого соседа w:
     if distance[v] + weight(v,w) < distance[w]:
       distance[w] = distance[v] + weight(v,w)
       parent[w] = v

4. Восстановить путь от start до end
```

### Сложность

O((V + E) log V) где V - вершины, E - ребра

### Когда использовать

- Нужен один лучший путь
- Скорость критична
- Датасет инструментов небольшой (< 100)

### Пример

```python
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm

finder = GraphStrategyFinder()

strategies = finder.find_strategies(
    tools=tools,
    start_format="text|question",
    end_format="text|answer",
    algorithm=PathfindingAlgorithm.DIJKSTRA,
    limit=1
)

# Результат: 1 кратчайший путь
print(f"Found: {len(strategies)} strategy")
print(f"Path: {[t.metadata.tool_name for t in strategies[0]]}")
```

---

## Алгоритм 2: A* (A-star)

### Назначение

Поиск с **эвристикой** для ускорения.

### Отличие от Dijkstra

```
Dijkstra: f(v) = g(v)  # Расстояние от start
A*:       f(v) = g(v) + h(v)  # Расстояние + эвристика до end
```

### Эвристика

```python
h(v) = оценка_расстояния_до_цели(v)
```

В GraphArchitect: h=0 (эквивалентно Dijkstra)

Можно улучшить: h = количество_шагов_до_цели

### Когда использовать

- Нужен один путь
- Есть хорошая эвристика
- Граф очень большой (> 1000 вершин)

---

## Алгоритм 3: Yen (K-shortest paths)

### Назначение

Найти **топ-K кратчайших путей**.

### Как работает

```
1. Найти кратчайший путь (Dijkstra) → Path_1

2. Для k = 2 до K:
   Для каждой вершины v в Path_{k-1}:
     - Удалить часть пути
     - Найти альтернативный путь (Dijkstra)
     - Добавить в кандидаты
   
   Выбрать лучший из кандидатов → Path_k

3. Вернуть [Path_1, Path_2, ..., Path_K]
```

### Сложность

O(K · V · (E + V log V))

Значительно медленнее Dijkstra, но дает альтернативы.

### Когда использовать

- Нужны альтернативные пути
- Хотите выбрать лучший из нескольких
- LLM может выбрать оптимальный вариант

### Пример

```python
strategies = finder.find_strategies(
    tools=tools,
    start_format="text|question",
    end_format="text|category",
    algorithm=PathfindingAlgorithm.YEN,
    limit=5  # Топ-5 путей
)

# Результат: До 5 разных путей
for i, strategy in enumerate(strategies, 1):
    path = [t.metadata.tool_name for t in strategy]
    weight = sum(t.get_graph_weight() for t in strategy)
    print(f"Strategy {i}: {' → '.join(path)} (weight: {weight:.3f})")
```

**Вывод**:
```
Strategy 1: GPT-4 Classifier (weight: 0.020)
Strategy 2: Claude Classifier (weight: 0.051)
Strategy 3: Local Classifier (weight: 0.248)
```

---

## Алгоритм 4: Ant Colony Optimization (ACO)

### Назначение

**Вероятностный поиск** топ-N путей с феромонами.

### Как работает

```
1. Инициализация феромонов на всех ребрах

2. Для каждой итерации:
   Для каждого муравья:
     - Строит путь вероятностно:
       P(ребро) ∝ (феромон^α) × (эвристика^β)
     - Запоминает путь и его качество
   
   Обновление феромонов:
     - Испарение: pheromone *= (1 - evaporation_rate)
     - Добавление: pheromone += 1/path_length (для хороших путей)

3. Вернуть лучшие найденные пути
```

### Параметры

```python
num_ants = 10           # Количество муравьев
num_iterations = 100    # Итераций
alpha = 1.0            # Вес феромона
beta = 2.0             # Вес эвристики
evaporation = 0.5      # Скорость испарения
```

### Когда использовать

- Очень большой граф (> 1000 вершин)
- Много похожих путей
- Хотите разнообразие вариантов
- Не критична оптимальность

### Пример

```python
strategies = finder.find_strategies(
    tools=tools,
    start_format="text|data",
    end_format="text|report",
    algorithm=PathfindingAlgorithm.ANT_COLONY,
    limit=10  # Топ-10
)

# Муравьи найдут разнообразные пути
```

---

## Сравнение алгоритмов

| Алгоритм | Скорость | Качество | Разнообразие | Использование |
|----------|----------|----------|--------------|---------------|
| Dijkstra | Быстро | Оптимален | Нет | Один лучший путь |
| A* | Очень быстро | Оптимален | Нет | Большие графы |
| Yen | Медленно | Топ-K оптимальных | Среднее | Альтернативы |
| ACO | Средне | Хорошие | Высокое | Разнообразие |

### Выбор алгоритма

**Для простых задач**:
```bash
planning_algorithm=dijkstra
```

**Для важных задач** (нужны альтернативы):
```bash
planning_algorithm=yen_5
```

**Для креативных задач**:
```bash
planning_algorithm=ant_10
```

---

## Практика

### Упражнение 1: Сравнение алгоритмов

Выполните один запрос с разными алгоритмами:

```python
algorithms = ["dijkstra", "yen_3", "yen_5", "ant_5"]

for algo in algorithms:
    response = requests.post(
        f"http://localhost:8000/api/chat/test/message/stream",
        data={
            "message": "Создать отчет по данным",
            "planning_algorithm": algo
        }
    )
    # Смотрите какие пути найдены
```

**Вопрос**: Какой алгоритм нашел больше стратегий?

### Упражнение 2: Измерение скорости

```python
import time

for algo in ["dijkstra", "yen_5", "ant_5"]:
    start = time.time()
    
    strategies = finder.find_strategies(
        tools, "text|question", "text|answer",
        PathfindingAlgorithm[algo.upper().split('_')[0]],
        limit=5
    )
    
    elapsed = time.time() - start
    print(f"{algo}: {elapsed*1000:.2f}ms, found {len(strategies)}")
```

---

## Итоги

### Вы узнали

- Dijkstra - быстрый, один путь
- A* - с эвристикой, еще быстрее
- Yen - топ-K путей, медленнее
- ACO - вероятностный, разнообразие
- Выбор зависит от задачи

### Когда что использовать

- Быстро и просто → Dijkstra
- Альтернативы → Yen
- Креативность → ACO
- Огромный граф → A*

---

**Следующий**: [Выбор инструментов](02_tool_selection.md)
