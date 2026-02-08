"""
Пример использования продвинутого селектора с полной формулой R(x).

Демонстрирует:
- Учет качества, стоимости и времени
- Формула из отчета: R(x) = (t_sc · (w_q · f_q - w_c · f_c)) / log₁₀(w_t · f_t + 1)
- Балансировка между критериями
"""

import sys
from pathlib import Path

# Добавляем GraphArchitect
grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

from grapharchitect.services.selection.advanced_instrument_selector import AdvancedInstrumentSelector
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector


class DemoTool(BaseTool):
    """Демо инструмент с разными параметрами."""
    
    def __init__(self, name, reputation, cost, time):
        super().__init__()
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
        self.metadata.mean_cost = cost
        self.metadata.mean_time_answer = time
        self.metadata.training_sample_size = 50
        self.metadata.variance_estimate = 0.1
        
        self.input = Connector("text", "question")
        self.output = Connector("text", "answer")
    
    def execute(self, input_data):
        return f"[{self.metadata.tool_name}] Processed"


print("=" * 70)
print("ПРИМЕР: Продвинутый селектор с формулой R(x)")
print("=" * 70)
print()

# Создаем инструменты с разными характеристиками
print("Шаг 1: Создание инструментов")
print("-" * 70)

tools = [
    #              Название           Rep   Cost   Time
    DemoTool("GPT-4 (Expensive)",    0.98,  0.030,  2.8),  # Дорогой, точный, медленный
    DemoTool("Claude (Balanced)",    0.95,  0.020,  3.2),  # Средний
    DemoTool("Local (Cheap)",        0.78,  0.001,  1.2),  # Дешевый, быстрый, менее точный
    DemoTool("Fast (Ultra-fast)",    0.72,  0.005,  0.8),  # Самый быстрый
]

for tool in tools:
    print(f"  {tool.metadata.tool_name:25} "
          f"rep={tool.metadata.reputation:.2f}  "
          f"cost=${tool.metadata.mean_cost:.3f}  "
          f"time={tool.metadata.mean_time_answer:.1f}s")

print()

# Создаем селектор
print("Шаг 2: Создание селектора")
print("-" * 70)

# Embedding service
embedding = SimpleEmbeddingService(dimension=384)
task_embedding = embedding.embed_text("Классифицировать текст")

# Генерируем эмбеддинги для инструментов
for tool in tools:
    tool.metadata.capabilities_embedding = embedding.embed_tool_capabilities(tool)

print()

# Тест 1: Приоритет качества
print("Тест 1: Приоритет КАЧЕСТВА")
print("-" * 70)

selector_quality = AdvancedInstrumentSelector(
    weight_quality=1.0,   # Высокий вес
    weight_cost=0.1,      # Низкий вес
    weight_time=0.1       # Низкий вес
)

result = selector_quality.select_instrument(tools, task_embedding, top_k=4)

print(f"  Выбран: {result.selected_tool.metadata.tool_name}")
print(f"  Вероятность: {result.selection_probability:.3f}")
print(f"  Температура: {result.temperature:.3f}")
print()

print("  Метрики R(x) для всех:")
for tool in tools:
    tool_id = id(tool)
    if tool_id in result.r_metrics:
        r_val = result.r_metrics[tool_id]
        prob = result.probabilities.get(tool_id, 0)
        print(f"    {tool.metadata.tool_name:25} R={r_val:7.3f}  P={prob:.3f}")

print()

# Тест 2: Приоритет стоимости
print("Тест 2: Приоритет СТОИМОСТИ")
print("-" * 70)

selector_cost = AdvancedInstrumentSelector(
    weight_quality=0.5,
    weight_cost=2.0,      # Высокий вес (минимизируем)
    weight_time=0.2
)

result = selector_cost.select_instrument(tools, task_embedding, top_k=4)

print(f"  Выбран: {result.selected_tool.metadata.tool_name}")
print(f"  Вероятность: {result.selection_probability:.3f}")
print()

print("  Метрики R(x) для всех:")
for tool in tools:
    tool_id = id(tool)
    if tool_id in result.r_metrics:
        r_val = result.r_metrics[tool_id]
        prob = result.probabilities.get(tool_id, 0)
        cost = result.costs[tool_id]
        print(f"    {tool.metadata.tool_name:25} R={r_val:7.3f}  P={prob:.3f}  cost=${cost:.3f}")

print()

# Тест 3: Приоритет скорости
print("Тест 3: Приоритет СКОРОСТИ")
print("-" * 70)

selector_speed = AdvancedInstrumentSelector(
    weight_quality=0.5,
    weight_cost=0.2,
    weight_time=2.0       # Высокий вес (минимизируем)
)

result = selector_speed.select_instrument(tools, task_embedding, top_k=4)

print(f"  Выбран: {result.selected_tool.metadata.tool_name}")
print(f"  Вероятность: {result.selection_probability:.3f}")
print()

print("  Метрики R(x) для всех:")
for tool in tools:
    tool_id = id(tool)
    if tool_id in result.r_metrics:
        r_val = result.r_metrics[tool_id]
        prob = result.probabilities.get(tool_id, 0)
        time_val = result.times[tool_id]
        print(f"    {tool.metadata.tool_name:25} R={r_val:7.3f}  P={prob:.3f}  time={time_val:.1f}s")

print()

# Тест 4: Сбалансированный
print("Тест 4: СБАЛАНСИРОВАННЫЙ")
print("-" * 70)

selector_balanced = AdvancedInstrumentSelector(
    weight_quality=1.0,
    weight_cost=0.5,
    weight_time=0.5
)

result = selector_balanced.select_instrument(tools, task_embedding, top_k=4)

print(f"  Выбран: {result.selected_tool.metadata.tool_name}")
print(f"  Вероятность: {result.selection_probability:.3f}")
print()

print("  Полные метрики:")
for tool in tools:
    tool_id = id(tool)
    if tool_id in result.r_metrics:
        print(f"  {tool.metadata.tool_name}:")
        print(f"    Логит (качество): {result.logits[tool_id]:.3f}")
        print(f"    Стоимость (norm): {result.costs[tool_id]:.3f}")
        print(f"    Время (norm):     {result.times[tool_id]:.3f}")
        print(f"    R(x):             {result.r_metrics[tool_id]:.3f}")
        print(f"    Вероятность:      {result.probabilities.get(tool_id, 0):.3f}")
        print()

# Итоги
print("=" * 70)
print("ИТОГИ")
print("=" * 70)
print()
print("Продвинутый селектор учитывает 3 фактора:")
print("  1. Качество (через логиты и репутацию)")
print("  2. Стоимость (API cost)")
print("  3. Время выполнения")
print()
print("Формула R(x) позволяет:")
print("  - Балансировать между критериями через веса")
print("  - Приоритизировать важные факторы")
print("  - Оптимизировать под конкретные требования")
print()
print("Применение:")
print("  - Production: минимизация стоимости")
print("  - Критичные задачи: максимум качества")
print("  - Real-time: минимум времени")
print()
print("=" * 70)
