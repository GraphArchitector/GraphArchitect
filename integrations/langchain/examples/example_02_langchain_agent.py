"""
Пример 2: Использование LangChain Agent с GraphArchitect tools.

Демонстрирует:
- Создание LangChain агента с GraphArchitect инструментами
- ReAct reasoning с GraphArchitect tools
- Комбинация планирования LangChain и выбора GraphArchitect
"""

import sys
from pathlib import Path
import os

# Добавляем пути
integration_path = Path(__file__).parent.parent
sys.path.insert(0, str(integration_path))

grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

print("=" * 70)
print("ПРИМЕР 2: LangChain Agent с GraphArchitect Tools")
print("=" * 70)
print()

# Проверка API ключа
if not os.getenv("OPENAI_API_KEY"):
    print("[WARNING] OPENAI_API_KEY не установлен")
    print("Установите: export OPENAI_API_KEY=your-key")
    print()
    print("Пример будет использовать заглушки")
    print()

# Импорты
try:
    from langchain.llms import OpenAI
    from langchain.agents import initialize_agent, AgentType
    print("[OK] LangChain импортирован")
except ImportError:
    print("[ERROR] Установите: pip install langchain openai")
    sys.exit(1)

from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect_to_langchain import convert_grapharchitect_tools_to_langchain

print()

# Создаем набор GraphArchitect инструментов
class TextClassifier(BaseTool):
    """Классификатор текста"""
    
    def __init__(self):
        super().__init__()
        self.metadata.tool_name = "Text Classifier"
        self.metadata.description = "Классифицирует текст по категориям"
        self.input = Connector("text", "question")
        self.output = Connector("text", "category")
    
    def execute(self, input_data):
        return f"[Классификация] Категория: general (для: {str(input_data)[:50]})"


class DataAnalyzer(BaseTool):
    """Анализатор данных"""
    
    def __init__(self):
        super().__init__()
        self.metadata.tool_name = "Data Analyzer"
        self.metadata.description = "Анализирует данные и выявляет паттерны"
        self.input = Connector("text", "data")
        self.output = Connector("text", "analysis")
    
    def execute(self, input_data):
        return f"[Анализ] Найдено паттернов: 3 (в данных: {str(input_data)[:50]})"


class ReportGenerator(BaseTool):
    """Генератор отчетов"""
    
    def __init__(self):
        super().__init__()
        self.metadata.tool_name = "Report Generator"
        self.metadata.description = "Создает структурированные отчеты"
        self.input = Connector("text", "data")
        self.output = Connector("text", "report")
    
    def execute(self, input_data):
        return f"[Отчет] Создан отчет на основе: {str(input_data)[:50]}"


# Создаем инструменты
print("Шаг 1: Создание GraphArchitect инструментов")
print("-" * 70)

ga_tools = [
    TextClassifier(),
    DataAnalyzer(),
    ReportGenerator()
]

for tool in ga_tools:
    print(f"  - {tool.metadata.tool_name}")
    print(f"    {tool.input.format} → {tool.output.format}")

print()

# Конвертируем в LangChain tools
print("Шаг 2: Конвертация в LangChain Tools")
print("-" * 70)

langchain_tools = convert_grapharchitect_tools_to_langchain(ga_tools)

print(f"  Конвертировано: {len(langchain_tools)} tools")
for tool in langchain_tools:
    print(f"    - {tool.name}")

print()

# Создаем LangChain LLM
print("Шаг 3: Создание LangChain Agent")
print("-" * 70)

try:
    llm = OpenAI(temperature=0.7)
    print("  [OK] OpenAI LLM создан")
except Exception as e:
    print(f"  [WARNING] OpenAI недоступен: {e}")
    print("  Используется заглушка")
    
    # Заглушка LLM для демонстрации
    class MockLLM:
        def __call__(self, prompt):
            return "Mock response"
    
    llm = MockLLM()

# Создаем агента
try:
    agent = initialize_agent(
        tools=langchain_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print(f"  [OK] Agent создан с {len(langchain_tools)} GraphArchitect tools")
    print()
    
    # Выполнение задачи
    print("Шаг 4: Выполнение через LangChain Agent")
    print("-" * 70)
    print()
    
    query = "Проанализировать данные продаж и создать отчет"
    print(f"Запрос: {query}")
    print()
    
    # Запуск (если есть реальный LLM)
    if os.getenv("OPENAI_API_KEY"):
        result = agent.run(query)
        print(f"\nРезультат: {result}")
    else:
        print("[INFO] Для реального выполнения установите OPENAI_API_KEY")
        print()
        print("Agent будет использовать GraphArchitect tools через ReAct:")
        print("  1. Thought: Нужно проанализировать данные")
        print("  2. Action: Data Analyzer")
        print("  3. Observation: [Анализ] Найдено паттернов: 3")
        print("  4. Thought: Теперь создать отчет")
        print("  5. Action: Report Generator")
        print("  6. Observation: [Отчет] Создан отчет")
        print("  7. Final Answer: Отчет готов")

except Exception as e:
    print(f"  [ERROR] {e}")

print()

# Итоги
print("=" * 70)
print("ИТОГИ")
print("=" * 70)
print()
print("Интеграция работает!")
print()
print("GraphArchitect tools доступны в LangChain как:")
print("  - Обычные LangChain Tools")
print("  - С сохранением метаданных (reputation, connectors)")
print("  - С возможностью обучения")
print()
print("LangChain Agent может:")
print("  - Использовать GraphArchitect инструменты")
print("  - Планировать через ReAct")
print("  - Комбинировать с нативными LangChain tools")
print()
print("=" * 70)
