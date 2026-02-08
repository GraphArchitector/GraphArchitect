"""
Пример 3: Гибридное выполнение.

Демонстрирует:
- Объединение GraphArchitect и LangChain tools
- Планирование через GraphArchitect граф
- Выполнение любых инструментов
- Обучение через Policy Gradient
"""

import sys
from pathlib import Path

# Добавляем пути
integration_path = Path(__file__).parent.parent
sys.path.insert(0, str(integration_path))

grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

print("=" * 70)
print("ПРИМЕР 3: Гибридное выполнение GraphArchitect + LangChain")
print("=" * 70)
print()

# Импорты GraphArchitect
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm

# Импорты LangChain
try:
    from langchain.tools import Tool
    print("[OK] LangChain доступен")
except ImportError:
    print("[ERROR] Установите: pip install langchain")
    sys.exit(1)

# Импорт гибридного исполнителя
from hybrid_executor import HybridExecutor

print()

# Создаем GraphArchitect инструменты
print("Шаг 1: Создание GraphArchitect инструментов")
print("-" * 70)


class Classifier(BaseTool):
    def __init__(self):
        super().__init__()
        self.metadata.tool_name = "GA Classifier"
        self.input = Connector("text", "question")
        self.output = Connector("text", "category")
    
    def execute(self, input_data):
        return f"[GA] Category: positive"


class Writer(BaseTool):
    def __init__(self):
        super().__init__()
        self.metadata.tool_name = "GA Writer"
        self.input = Connector("text", "outline")
        self.output = Connector("text", "content")
    
    def execute(self, input_data):
        return f"[GA] Content created from outline"


ga_tools = [Classifier(), Writer()]

for tool in ga_tools:
    print(f"  {tool.metadata.tool_name}: {tool.input.format} → {tool.output.format}")

print()

# Создаем LangChain инструменты
print("Шаг 2: Создание LangChain инструментов")
print("-" * 70)

def search(query):
    return f"[LC] Search results for: {query}"

def summarize(text):
    return f"[LC] Summary of: {text[:50]}..."

lc_tools = [
    Tool(name="Search", description="Searches information", func=search),
    Tool(name="Summarizer", description="Summarizes text", func=summarize)
]

for tool in lc_tools:
    print(f"  {tool.name}: {tool.description}")

print()

# Создаем гибридный исполнитель
print("Шаг 3: Инициализация HybridExecutor")
print("-" * 70)

executor = HybridExecutor()

# Добавляем GraphArchitect tools
executor.add_grapharchitect_tools(ga_tools)

# Добавляем LangChain tools с указанием коннекторов
executor.add_langchain_tools(
    lc_tools,
    connector_mappings={
        "Search": (Connector("text", "query"), Connector("text", "findings")),
        "Summarizer": (Connector("text", "document"), Connector("text", "summary"))
    }
)

summary = executor.get_tools_summary()
print(f"  [OK] Инициализирован с {summary['total_tools']} инструментами")
print(f"    GraphArchitect sources: {summary['grapharchitect_tools']}")
print(f"    LangChain sources: {summary['langchain_tools']}")
print()

# Выполнение задачи
print("Шаг 4: Выполнение гибридной задачи")
print("-" * 70)
print()

print("Задача: Классифицировать запрос клиента")
print()

# Выполнение через гибридный executor
# GraphArchitect использует ВСЕ инструменты (и свои, и LangChain)
context = executor.execute_task(
    description="Классифицировать запрос клиента",
    input_data="Жалоба на задержку доставки",
    input_connector=Connector("text", "question"),
    output_connector=Connector("text", "category"),
    algorithm=PathfindingAlgorithm.DIJKSTRA,
    path_limit=1,
    top_k=3
)

print(f"Статус: {context.status.value}")
print(f"Результат: {context.result}")
print(f"Шагов: {context.get_total_steps()}")
print(f"Время: {context.total_time:.3f}s")
print()

print("Использованные инструменты:")
for i, step in enumerate(context.execution_steps, 1):
    print(f"  {i}. {step.selected_tool.metadata.tool_name}")
    print(f"     Вероятность: {step.selection_result.selection_probability:.3f}")

print()

# Итоги
print("=" * 70)
print("ИТОГИ")
print("=" * 70)
print()
print("Гибридный executor объединяет лучшее из двух миров:")
print()
print("От GraphArchitect:")
print("  - Планирование на основе графа")
print("  - Softmax выбор инструментов")
print("  - Адаптивная температура")
print("  - Policy Gradient обучение")
print()
print("От LangChain:")
print("  - Богатая экосистема tools")
print("  - Интеграции с API")
print("  - Chains и Agents")
print("  - Community поддержка")
print()
print("Результат:")
print("  - Любые LangChain tools в GraphArchitect")
print("  - Интеллектуальное планирование и выбор")
print("  - Обучение на результатах")
print()
print("=" * 70)
