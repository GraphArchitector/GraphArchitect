# RLAIF - Reinforcement Learning from AI Feedback


RLAIF (Reinforcement Learning from AI Feedback) - это подход к обучению систем, где вместо оценок от человека используются оценки от другой AI модели (LLM судья). Эти автоматические оценки служат обратной связью для обновления и улучшения инструментов в системе.

## Основная идея

```
Выполнение задачи → LLM Critic оценивает результат → Обновление репутации инструментов → Лучшие результаты
```

Система работает по циклу:
1. Инструменты выполняют задачу
2. LLM критик оценивает качество результата
3. Инструменты с хорошими оценками повышают репутацию
4. При следующем выборе инструмента система отдает предпочтение более "успешным" инструментам

## Компоненты системы

### LLMCritic - Судья качества

LLMCritic - это LLM (большая языковая модель), которая выступает в роли эксперта-оценщика. Она анализирует результаты работы системы и выставляет баллы по четырем независимым критериям:

**Критерии оценки:**
- **Correctness (Правильность)** - правильность решения задачи, отсутствие фактических ошибок
- **Completeness (Полнота)** - полнота покрытия всех аспектов задачи
- **Relevance (Релевантность)** - соответствие результата исходной задаче
- **Clarity (Ясность)** - понятность и структурированность результата

Каждый критерий оценивается от 0 до 10, затем нормализуется в диапазон 0-1.

**Поддерживаемые бэкенды:**
- **OpenRouter** - облачный API с доступом к GPT-4, Claude, Gemini и другим моделям
- **VLLM** - локальный запуск открытых моделей (Qwen, Llama, Mistral)

**Примеры процесса оценки:**

Для задачи "Классифицировать тональность текста":
- Исходный текст: "Этот фильм просто восхитителен!"
- Результат системы: "[SENTIMENT] positive"
- Оценка критика:
  - Correctness: 10/10 (правильно определена тональность)
  - Completeness: 8/10 (ответ полный, но нет объяснения)
  - Relevance: 10/10 (прямое ответ на вопрос)
  - Clarity: 9/10 (ясно, но формат можно улучшить)
  - Overall: 9.25/10

### RLAIFTrainer - Оркестратор обучения

RLAIFTrainer координирует весь процесс обучения:

```
ExecutionContext → Оценка (LLMCritic) → Feedback → Обновление инструментов → Новые репутации
```

**Полный цикл обучения:**

1. **Оценка результата** - результат выполнения передается LLMCritic
2. **Валидация оценки** - проверяется корректность полученных баллов
3. **Создание FeedbackData** - оценка преобразуется в структурированную обратную связь
4. **Обновление репутации** - репутация инструментов обновляется в соответствии с оценкой
5. **Сохранение истории** - все метрики записываются для анализа

## Механика обновления репутации

Когда инструмент получает оценку, его репутация обновляется. Система поддерживает три стратегии обновления:

### Policy Gradient (рекомендуется по умолчанию)

```
delta = learning_rate × reward
new_reputation = old_reputation + delta
```

**Пример:**
- Старая репутация инструмента: 0.5
- Learning rate: 0.01
- Reward (оценка): 0.8
- Delta: 0.01 × 0.8 = 0.008
- Новая репутация: 0.5 + 0.008 = 0.508

Используется для постепенного, контролируемого улучшения. Learning rate определяет скорость обучения.

### Exponential Moving Average (EMA)

```
new_reputation = alpha × reward + (1 - alpha) × old_reputation
```

**Пример:**
- Старая репутация: 0.5
- Alpha: 0.1
- Reward: 0.8
- Новая репутация: 0.1 × 0.8 + 0.9 × 0.5 = 0.08 + 0.45 = 0.53

Используется для сглаживания колебаний оценок. Альфа определяет, насколько быстро "забываются" старые значения.

### Direct Assignment

```
new_reputation = reward
```

Используется при полном переобучении, когда старые значения репутации считаются неверными.

## Жизненный цикл инструмента

```
Инструмент создается
    ↓
Репутация = 0.5 (нейтральное значение)
    ↓
Используется в задачах → получает оценки
    ↓
За хорошие оценки репутация растет (до 1.0)
За плохие оценки репутация падает (до 0.0)
    ↓
Инструменты с высокой репутацией выбираются чаще
Инструменты с низкой репутацией выбираются реже
```

## Отслеживание прогресса обучения

### Статистика оценок

По мере обучения система собирает статистику:

```python
stats = trainer.get_evaluation_statistics()

# Результат:
{
    'total_evaluations': 50,              # Всего оценок
    'overall': {
        'mean': 0.75,                     # Средняя оценка
        'stdev': 0.12,                    # Стандартное отклонение
        'min': 0.45,                      # Минимум
        'max': 0.95,                      # Максимум
        'median': 0.76                    # Медиана
    },
    'correctness': {...},                 # По каждому критерию
    'completeness': {...},
    'relevance': {...},
    'clarity': {...}
}
```

### Метрики обучения

Система отслеживает, как улучшается качество со временем:

```python
metrics = trainer.get_learning_metrics()

# Результат:
{
    'iterations': 50,                     # Всего итераций
    'initial_score': 0.60,                # Начальный балл
    'final_score': 0.85,                  # Финальный балл
    'total_improvement': 0.25,            # Абсолютное улучшение
    'improvement_percent': 41.67,         # Процентное улучшение
    'convergence_rate': 0.0012,           # Скорость сходимости
    'is_converged': True,                 # Система сошлась?
    'tools_updated_total': 152            # Всего обновлений
}
```

### История репутации инструментов

Система ведет полную историю изменений репутации каждого инструмента:

```python
dynamics = trainer.get_reputation_dynamics("Classifier")

# Результат:
{
    'tool_name': 'Classifier',
    'initial_reputation': 0.50,           # Начальная репутация
    'final_reputation': 0.78,             # Финальная репутация
    'total_change': 0.28,                 # Общее изменение
    'max_reputation': 0.82,               # Максимум во времени
    'min_reputation': 0.45,               # Минимум во времени
    'mean_reputation': 0.65,              # Средняя репутация
    'total_updates': 50,                  # Количество обновлений
    'mean_change_per_update': 0.0056      # Среднее изменение
}
```

## Проверка сходимости

По мере обучения система должна "сойтись" - оценки перестают меняться и стабилизируются:

```python
# Простая проверка
if trainer.check_convergence():
    print("Система сошлась - обучение завершено")
```

Сходимость определяется как: максимальное изменение оценок в последних N итерациях меньше порога.

**Параметры сходимости:**
- `convergence_window_size` - сколько последних итераций анализировать (по умолчанию 5)
- `convergence_tolerance` - допустимое изменение между итерациями (по умолчанию 0.001)

**Пример сходящейся системы:**

```
Итерация 1: 0.60
Итерация 2: 0.65 (изменение: 0.05)
Итерация 3: 0.70 (изменение: 0.05)
Итерация 4: 0.74 (изменение: 0.04)
Итерация 5: 0.77 (изменение: 0.03)
Итерация 6: 0.79 (изменение: 0.02)
Итерация 7: 0.80 (изменение: 0.01)
Итерация 8: 0.805 (изменение: 0.005) ← максимальное изменение < 0.001?
```

## Batch обработка и масштабируемость

Система поддерживает обработку нескольких задач за раз:

```python
executions = [
    (context1, "Task 1", "Answer 1"),
    (context2, "Task 2", "Answer 2"),
    (context3, "Task 3", "Answer 3"),
    # ... до тысяч задач
]

result = trainer.batch_evaluate_and_train(executions)
```

**Что происходит при batch обработке:**
1. Каждая задача независимо оценивается LLMCritic
2. Если какая-то оценка не удалась - она логируется и пропускается
3. Результаты агрегируются (усредняются)
4. Возвращается единый отчет обо всех выполнениях

**Результат batch обучения:**
```python
{
    'evaluations_count': 100,          # Всего попыток
    'successful_count': 98,            # Успешных
    'failed_count': 2,                 # Ошибок
    'average_score': 0.76,             # Средняя оценка
    'average_correctness': 0.75,
    'average_completeness': 0.74,
    'average_relevance': 0.78,
    'average_clarity': 0.77,
    'tools_updated': 245,              # Всего обновлений
    'improvements': {                  # Средние изменения
        'Classifier': 0.012,
        'Parser': 0.008,
        'Writer': 0.005
    }
}
```

## Consensus оценивание

Для повышения надежности результатов можно использовать несколько независимых судей:

```python
# Одна оценка от одного судьи
score = trainer.evaluate_and_train(context, task, result)

# Три оценки от трех независимых экземпляров критика
consensus_score = trainer.evaluate_with_consensus(
    task="Классифицировать текст",
    answer="[SENTIMENT] positive",
    num_evaluators=3
)
```

**Как это работает:**
1. Задача оценивается тремя разными экземплярами LLMCritic (или одним, но несколько раз)
2. Получается три независимых оценки
3. Результаты усредняются
4. Возвращается более надежная consensus оценка

**Пример:**
```
Судья 1: correctness=0.9, overall=0.85
Судья 2: correctness=0.85, overall=0.82
Судья 3: correctness=0.92, overall=0.87
---
Consensus: correctness=0.89, overall=0.85
```

## Конфигурация и воспроизводимость

Вся конфигурация сохраняется для обеспечения воспроизводимости:

```python
from grapharchitect.services.rlaif import RLAIFTrainingConfig

config = RLAIFTrainingConfig(
    # Обучение
    learning_rate=0.01,                   # Скорость обучения
    update_strategy=UpdateStrategy.POLICY_GRADIENT,  # Стратегия обновления
    min_score_threshold=0.3,              # Минимум для успеха
    
    # Сглаживание (для EMA)
    ema_alpha=0.1,                        # Альфа для EMA
    
    # Мониторинг сходимости
    convergence_window_size=5,            # Окно для проверки
    convergence_tolerance=0.001,          # Допуск
    
    # Сохранение
    save_evaluations=True,                # Сохранять оценки
    save_tool_updates=True,               # Сохранять обновления
    
    # История
    max_reputation_history=1000,          # Максимум записей
    
    # Логирование
    log_level="INFO",
    seed=42                               # Для воспроизводимости
)
```

**Что сохраняется в конфигурацию:**
- Все параметры обучения
- Версия промпта критика
- Версия модели критика
- Метаданные времени выполнения
- ID каждой оценки (для отслеживания)

## Сохранение результатов обучения

Система может сохранить полные логи обучения для анализа:

```python
trainer.save_training_log("training_log.json")
```

**Что содержится в логе:**
```json
{
    "metadata": {
        "saved_at": "2024-01-15T10:30:00",
        "config": { ... },
        "total_evaluations": 50
    },
    "learning_metrics": {
        "iterations": 50,
        "improvement_percent": 41.67
    },
    "evaluations": [
        {
            "evaluation_id": "uuid",
            "timestamp": "2024-01-15T10:00:00",
            "overall": 0.75,
            "correctness": 0.74,
            "reasoning": "Good answer but missing details"
        }
    ],
    "tool_updates": [
        {
            "tool_name": "Classifier",
            "old_reputation": 0.50,
            "new_reputation": 0.508,
            "delta_reputation": 0.008
        }
    ],
    "reputation_timeline": {
        "Classifier": [
            {"timestamp": "...", "reputation": 0.50},
            {"timestamp": "...", "reputation": 0.508}
        ]
    }
}
```

## Сравнение систем

Система может сравнить результаты обучения двух разных тренеров:

```python
trainer_a = RLAIFTrainer(critic, orchestrator)
trainer_b = RLAIFTrainer(critic, orchestrator)

# Обучение обеих систем на одинаковых задачах...

comparison = trainer_a.compare_systems(
    trainer_b,
    system_a_name="GraphArchitect",
    system_b_name="AutoGen"
)
```

**Результат сравнения:**
```python
{
    'system_a': {
        'name': 'GraphArchitect',
        'overall_mean': 0.81,
        'correctness_mean': 0.80,
        'improvement_percent': 35.0
    },
    'system_b': {
        'name': 'AutoGen',
        'overall_mean': 0.61,
        'correctness_mean': 0.59,
        'improvement_percent': 1.7
    },
    'comparison': {
        'overall_improvement_percent': 32.8,  # На сколько A лучше B
        'correctness_improvement_percent': 35.6,
        'both_converged': True
    }
}
```

## Обработка ошибок

Система спроектирована для устойчивости к ошибкам:

- Если LLMCritic не отвечает - возвращается fallback оценка
- Если JSON парсинг не удался - логируется и пропускается
- Если инструмент не обновился - продолжается со следующего
- Если batch имеет некорректные элементы - они пропускаются
- Если критик вернул невалидную оценку - проверяется диапазон

Все ошибки логируются на разных уровнях для отладки.

## Интеграция с системой

RLAIF может быть интегрирован в основной цикл выполнения:

```python
# После выполнения задачи
context = orchestrator.execute_task(task, tools)

# Автоматическая оценка и обучение
if rlaif_trainer:
    result = rlaif_trainer.evaluate_and_train(
        context,
        task.description,
        context.result
    )
    
    if result:
        print(f"Обучено инструментов: {result.tools_updated}")
```

## Стоимость и производительность

### OpenRouter (облачный API)

| Модель | Стоимость | Время | Примечание |
|--------|-----------|-------|-----------|
| GPT-3.5-turbo | $0.001 за оценку | 1-3s | Экономичный вариант |
| GPT-4 | $0.015 за оценку | 2-5s | Лучшее качество |
| Claude Sonnet | $0.002 за оценку | 1-4s | Сбалансированный |

Для 1000 оценок:
- GPT-3.5-turbo: $1
- GPT-4: $15
- Claude: $2

### VLLM (локально)

```bash
# Запуск локальной модели
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Стоимость: $0 (только электричество + GPU)
Скорость: 0.5-2s за оценку (зависит от GPU)
Privacy: Полная (локально)

## Примеры использования

### Базовое использование

```python
from grapharchitect.services.rlaif import LLMCritic, RLAIFTrainer

# Создание критика
critic = LLMCritic(
    backend="openrouter",
    model_name="openai/gpt-4",
    temperature=0.2
)

# Создание тренера
trainer = RLAIFTrainer(
    llm_critic=critic,
    training_orchestrator=training_orchestrator
)

# Обучение на одной задаче
result = trainer.evaluate_and_train(
    context=execution_context,
    task_description="Классифицировать тональность",
    result="[SENTIMENT] positive"
)
```

### Batch обучение

```python
# Несколько задач
tasks = [
    (context1, "Task 1", "Answer 1"),
    (context2, "Task 2", "Answer 2"),
    (context3, "Task 3", "Answer 3"),
]

# Batch обработка
result = trainer.batch_evaluate_and_train(tasks)

print(f"Успешно: {result.successful_count}/{result.evaluations_count}")
print(f"Средний балл: {result.average_score:.2f}")
```

### Анализ результатов

```python
# Получить метрики
metrics = trainer.get_learning_metrics()
stats = trainer.get_evaluation_statistics()

# Проверить сходимость
if trainer.check_convergence():
    print("Обучение завершено")

# Сохранить результаты
trainer.save_training_log("results.json")
```