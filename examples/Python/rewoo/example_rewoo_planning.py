"""
Пример использования ReWOO (Reasoning Without Observation).

Демонстрирует:
- Поиск нескольких стратегий через Yen
- Создание детального плана через Gemini 3 Flash
- Анализ и оптимизация плана
- Выполнение по плану
"""

import sys
from pathlib import Path
import os

# Добавляем GraphArchitect
grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

print("=" * 70)
print("ПРИМЕР: ReWOO Planning")
print("=" * 70)
print()

# Проверка API ключа
HAS_API_KEY = bool(os.getenv('OPENROUTER_API_KEY'))

if not HAS_API_KEY:
    print("[WARNING] OPENROUTER_API_KEY не установлен")
    print("ReWOO требует API ключ для Gemini/GPT")
    print()
    print("Установите: set OPENROUTER_API_KEY=your-key")
    print()
    print("Пример покажет структуру без реальных вызовов")
    print()

# Импорты
from grapharchitect.planning.rewoo_planner import ReWOOPlanner
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector


# Создаем демо инструменты
class DemoTool(BaseTool):
    """Демонстрационный инструмент."""
    
    def __init__(self, name, input_fmt, output_fmt, reputation):
        super().__init__()
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
        
        inp = input_fmt.split("|")
        out = output_fmt.split("|")
        
        self.input = Connector(inp[0], inp[1])
        self.output = Connector(out[0], out[1])
    
    def execute(self, input_data):
        return f"[{self.metadata.tool_name}] Success"


print("Шаг 1: Создание инструментов для графа")
print("-" * 70)

# Создаем граф с несколькими путями
tools = [
    # Путь 1 (прямой): Classifier → Responder
    DemoTool("Classifier-A", "text|question", "text|category", 0.90),
    DemoTool("Responder-A", "text|category", "text|response", 0.88),
    
    # Путь 2 (через анализ): Analyzer → Generator
    DemoTool("Analyzer-B", "text|question", "text|analysis", 0.92),
    DemoTool("Generator-B", "text|analysis", "text|response", 0.85),
    
    # Путь 3 (через outline): Outliner → Writer
    DemoTool("Outliner-C", "text|question", "text|outline", 0.87),
    DemoTool("Writer-C", "text|outline", "text|response", 0.89),
]

# Генерация эмбеддингов
embedding = SimpleEmbeddingService(dimension=384)
for tool in tools:
    tool.metadata.capabilities_embedding = embedding.embed_tool_capabilities(tool)

print(f"  Создано инструментов: {len(tools)}")
print()

# Шаг 2: Поиск стратегий
print("Шаг 2: Поиск стратегий через Yen algorithm")
print("-" * 70)

finder = GraphStrategyFinder()

strategies = finder.find_strategies(
    tools=tools,
    start_format="text|question",
    end_format="text|response",
    algorithm=PathfindingAlgorithm.YEN,
    limit=5
)

print(f"  Найдено стратегий: {len(strategies)}")
print()

for i, strategy in enumerate(strategies, 1):
    tool_names = [t.metadata.tool_name for t in strategy]
    path_str = " → ".join(tool_names)
    print(f"  Стратегия {i}: {path_str}")

print()

# Шаг 3: ReWOO Planning
print("Шаг 3: Создание детального плана (ReWOO)")
print("-" * 70)

if HAS_API_KEY:
    try:
        # Создаем ReWOO планировщик
        planner = ReWOOPlanner(
            gemini_api_key=os.getenv('OPENROUTER_API_KEY')
        )
        
        print("  [OK] ReWOO Planner инициализирован (Gemini 1.5 Flash)")
        print()
        
        # Создаем план
        print("  Отправка запроса к Gemini...")
        
        plan = planner.create_plan(
            task_description="Ответить на вопрос клиента о продукте",
            strategies=strategies,
            algorithm_used="yen_5"
        )
        
        if plan:
            print(f"\n  [OK] План создан!")
            print()
            print(f"  Обоснование LLM:")
            print(f"    {plan.reasoning[:200]}...")
            print()
            print(f"  План выполнения ({len(plan.steps)} шагов):")
            
            for i, step in enumerate(plan.steps, 1):
                print(f"\n    Шаг {i}: {step.tool_name}")
                print(f"      Описание: {step.description}")
                print(f"      Зависит от: {step.depends_on if step.depends_on else 'нет'}")
                print(f"      Ожидаемый результат: {step.expected_output}")
            
            print()
            print(f"  Оценки:")
            print(f"    Время: {plan.estimated_time:.1f}s")
            print(f"    Стоимость: ${plan.estimated_cost:.4f}")
        else:
            print("\n  [ERROR] Не удалось создать план")
    
    except Exception as e:
        print(f"\n  [ERROR] Ошибка ReWOO: {e}")
        import traceback
        traceback.print_exc()

else:
    print("  [SKIP] Для реального теста установите OPENROUTER_API_KEY")
    print()
    print("  Ожидаемый план (пример):")
    print()
    print("  Обоснование LLM:")
    print("    'Выбираю стратегию 1 (Classifier → Responder) как наиболее")
    print("    эффективную: прямая, быстрая, высокая точность классификатора...'")
    print()
    print("  План выполнения (2 шага):")
    print()
    print("    Шаг 1: Classifier-A")
    print("      Описание: Classify customer question into category")
    print("      Зависит от: нет")
    print("      Ожидаемый результат: Category label")
    print()
    print("    Шаг 2: Responder-A")
    print("      Описание: Generate response based on category")
    print("      Зависит от: ['step-1']")
    print("      Ожидаемый результат: Customer response")
    print()
    print("  Оценки:")
    print("    Время: 3.5s")
    print("    Стоимость: $0.045")

print()

# Итоги
print("=" * 70)
print("ИТОГИ")
print("=" * 70)
print()
print("ReWOO подход:")
print("  1. Граф алгоритм находит множество цепочек (Yen топ-5)")
print("  2. LLM (Gemini 3 Flash) анализирует все цепочки")
print("  3. Создается оптимальный детальный план")
print("  4. План выполняется без промежуточных LLM вызовов")
print()
print("Преимущества vs стандартный подход:")
print("  - LLM видит ВСЕ альтернативы сразу")
print("  - Может выбрать оптимальную или комбинировать")
print("  - Обоснование выбора")
print("  - Оценка времени и стоимости")
print("  - Один LLM вызов вместо множества")
print()
print("Применение:")
print("  - Сложные многошаговые задачи")
print("  - Когда важна оптимизация")
print("  - Production системы (меньше LLM вызовов)")
print()
print("Для использования:")
print("  1. Установите OPENROUTER_API_KEY")
print("  2. В Web UI поставьте галочку 'ReWOO Planning'")
print("  3. Или через API: use_rewoo=true")
print()
print("=" * 70)
