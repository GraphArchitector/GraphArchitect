"""
Пример 1: Базовая интеграция GraphArchitect с LangChain.

Демонстрирует:
- Использование GraphArchitect tools в LangChain
- Использование LangChain tools в GraphArchitect
- Гибридное выполнение
"""

import sys
from pathlib import Path

# Добавляем пути
integration_path = Path(__file__).parent.parent
sys.path.insert(0, str(integration_path))

grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

print("=" * 70)
print("ПРИМЕР 1: Базовая интеграция GraphArchitect + LangChain")
print("=" * 70)
print()

# Проверка зависимостей
try:
    from langchain.llms import OpenAI
    from langchain.tools import Tool
    print("[OK] LangChain импортирован")
except ImportError:
    print("[ERROR] LangChain не установлен")
    print("Установите: pip install langchain openai")
    sys.exit(1)

try:
    from grapharchitect.entities.base_tool import BaseTool
    from grapharchitect.entities.connectors.connector import Connector
    print("[OK] GraphArchitect импортирован")
except ImportError:
    print("[ERROR] GraphArchitect не доступен")
    sys.exit(1)

# Импорт адаптеров
from grapharchitect_to_langchain import GraphArchitectToolWrapper, convert_grapharchitect_tools_to_langchain
from langchain_to_grapharchitect import LangChainToolWrapper, convert_langchain_tools_to_grapharchitect

print()


# Создаем простой GraphArchitect инструмент
class SimpleClassifier(BaseTool):
    """Простой классификатор для демонстрации"""
    
    def __init__(self):
        super().__init__()
        self.metadata.tool_name = "Simple Classifier"
        self.metadata.description = "Classifies text into categories"
        self.metadata.reputation = 0.85
        
        self.input = Connector("text", "question")
        self.output = Connector("text", "category")
    
    def execute(self, input_data):
        # Простая логика
        text = str(input_data).lower()
        
        if "жалоба" in text or "проблема" in text:
            return "complaint"
        elif "вопрос" in text or "как" in text:
            return "question"
        else:
            return "other"


# Создаем LangChain инструмент
def simple_search(query: str) -> str:
    """Простой поиск (заглушка)"""
    return f"Search results for: {query}"

langchain_tool = Tool(
    name="Simple Search",
    description="Searches for information",
    func=simple_search
)

print("Шаг 1: Создание инструментов")
print("-" * 70)

ga_tool = SimpleClassifier()
print(f"  GraphArchitect tool: {ga_tool.metadata.tool_name}")
print(f"    {ga_tool.input.format} → {ga_tool.output.format}")

print(f"  LangChain tool: {langchain_tool.name}")
print()

# Тест 1: GraphArchitect tool в LangChain
print("Шаг 2: GraphArchitect tool → LangChain")
print("-" * 70)

wrapped_for_langchain = GraphArchitectToolWrapper(grapharchitect_tool=ga_tool)

print(f"  Обернутый tool:")
print(f"    Name: {wrapped_for_langchain.name}")
print(f"    Description: {wrapped_for_langchain.description[:100]}...")

# Выполнение
result = wrapped_for_langchain.run("Жалоба на задержку доставки")
print(f"    Result: {result}")
print()

# Тест 2: LangChain tool в GraphArchitect
print("Шаг 3: LangChain tool → GraphArchitect")
print("-" * 70)

wrapped_for_ga = LangChainToolWrapper(
    langchain_tool=langchain_tool,
    input_connector=Connector("text", "query"),
    output_connector=Connector("text", "findings")
)

print(f"  Обернутый tool:")
print(f"    Name: {wrapped_for_ga.metadata.tool_name}")
print(f"    {wrapped_for_ga.input.format} → {wrapped_for_ga.output.format}")

# Выполнение
result = wrapped_for_ga.execute("machine learning")
print(f"    Result: {result}")
print()

# Тест 3: Гибридное использование
print("Шаг 4: Гибридное выполнение")
print("-" * 70)

from hybrid_executor import HybridExecutor

# Создаем гибридный исполнитель
executor = HybridExecutor()

# Добавляем GraphArchitect tools
executor.add_grapharchitect_tools([ga_tool])

# Добавляем LangChain tools (они конвертируются автоматически)
executor.add_langchain_tools(
    [langchain_tool],
    connector_mappings={
        "Simple Search": (
            Connector("text", "query"),
            Connector("text", "findings")
        )
    }
)

# Сводка
summary = executor.get_tools_summary()
print(f"  Всего инструментов: {summary['total_tools']}")
print(f"    GraphArchitect: {summary['grapharchitect_tools']}")
print(f"    LangChain: {summary['langchain_tools']}")
print()

print("  Список инструментов:")
for tool_info in summary['tools']:
    print(f"    - {tool_info['name']} ({tool_info['source']})")
    print(f"      {tool_info['input']} → {tool_info['output']}")

print()

# Итоги
print("=" * 70)
print("ИТОГИ")
print("=" * 70)
print()
print("Успешно продемонстрировано:")
print("  1. GraphArchitect tools можно использовать в LangChain")
print("  2. LangChain tools можно использовать в GraphArchitect")
print("  3. Гибридный executor объединяет оба подхода")
print()
print("Преимущества:")
print("  - Используйте любые LangChain tools в GraphArchitect")
print("  - Планирование графа для LangChain tools")
print("  - Softmax выбор для любых инструментов")
print("  - Обучение через Policy Gradient")
print()
print("=" * 70)
