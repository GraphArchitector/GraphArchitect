#!/usr/bin/env python
"""
Проверка работоспособности примеров GraphArchitect.

Запускает базовые проверки перед выполнением примеров.
"""

import sys
from pathlib import Path

# Добавляем путь к библиотеке
project_root = Path(__file__).parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(project_root))


def check_imports():
    """Проверка импортов"""
    print("\n" + "="*70)
    print("1️⃣  ПРОВЕРКА ИМПОРТОВ")
    print("="*70)
    
    try:
        import grapharchitect
        print(f"✅ grapharchitect импортирован")
        print(f"   Версия: {grapharchitect.__version__}")
        
        from grapharchitect.entities import BaseTool
        print(f"✅ BaseTool импортирован")
        
        from grapharchitect.services import ExecutionOrchestrator
        print(f"✅ ExecutionOrchestrator импортирован")
        
        from grapharchitect.algorithms.pathfinding import Dijkstra
        print(f"✅ Dijkstra импортирован")
        
        return True
    
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("\n💡 Решение:")
        print(f"   export PYTHONPATH='{project_root}:$PYTHONPATH'")
        return False


def check_reactive_tool():
    """Проверка ReactiveTool"""
    print("\n" + "="*70)
    print("2️⃣  ПРОВЕРКА REACTIVE_TOOL")
    print("="*70)
    
    try:
        from reactive_tool import ReactiveTool
        
        # Создаем тестовый инструмент
        tool = ReactiveTool(
            "text", "input",
            "text", "output",
            "TestTool",
            lambda x: f"Processed: {x}"
        )
        
        # Проверяем выполнение
        result = tool.execute("test")
        
        print(f"✅ ReactiveTool работает")
        print(f"   Результат: {result}")
        
        return True
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def check_simple_execution():
    """Проверка простого выполнения"""
    print("\n" + "="*70)
    print("3️⃣  ПРОВЕРКА ВЫПОЛНЕНИЯ")
    print("="*70)
    
    try:
        from grapharchitect.entities import BaseTool, TaskDefinition
        from grapharchitect.entities.connectors import Connector
        from grapharchitect.services.embedding import SimpleEmbeddingService
        from grapharchitect.services.selection import InstrumentSelector
        from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
        from grapharchitect.services.execution import ExecutionOrchestrator
        
        # Простой инструмент
        class TestTool(BaseTool):
            def __init__(self):
                super().__init__()
                self.input = Connector("text", "input")
                self.output = Connector("text", "output")
                self.metadata.tool_name = "Test"
            
            def execute(self, data):
                return f"Result: {data}"
        
        # Инициализация
        embedding = SimpleEmbeddingService()
        selector = InstrumentSelector()
        finder = GraphStrategyFinder()
        orchestrator = ExecutionOrchestrator(embedding, selector, finder)
        
        # Задача
        task = TaskDefinition(
            description="Test",
            input_connector=Connector("text", "input"),
            output_connector=Connector("text", "output"),
            input_data="test"
        )
        
        # Выполнение
        context = orchestrator.execute_task(task, [TestTool()], path_limit=1)
        
        print(f"✅ Выполнение работает")
        print(f"   Статус: {context.status.value}")
        print(f"   Результат: {context.result}")
        
        return context.status.value == "completed"
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запуск всех проверок"""
    print("\n" + "="*70)
    print("🔍 ПРОВЕРКА ПРИМЕРОВ GRAPHARCHITECT")
    print("="*70)
    
    results = []
    results.append(("Импорты", check_imports()))
    results.append(("ReactiveTool", check_reactive_tool()))
    results.append(("Выполнение", check_simple_execution()))
    
    # Итоги
    print("\n" + "="*70)
    print("📊 ИТОГИ")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print("\n" + "="*70)
    
    if passed_count == total_count:
        print(f"🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ({passed_count}/{total_count})")
        print("="*70)
        print("\n✅ Примеры готовы к запуску!")
        print("\n🚀 Запустите: python pathfind_test.py")
        return 0
    else:
        print(f"⚠️ ПРОЙДЕНО: {passed_count}/{total_count}")
        print("="*70)
        print("\n❌ Некоторые проверки не прошли")
        print("\n💡 Проверьте установку и пути импорта")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
