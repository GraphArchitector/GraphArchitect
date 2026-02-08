"""
Простой пример использования GraphArchitect.

Демонстрирует минимальный код для:
- Создания инструментов
- Выполнения задачи
- Получения результата
"""

import sys
from pathlib import Path

# Добавляем путь к библиотеке
project_root = Path(__file__).parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(project_root))

from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.task_definition import TaskDefinition
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.selection.instrument_selector import InstrumentSelector
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.execution.execution_orchestrator import ExecutionOrchestrator


# ==================== Простой инструмент ====================

class SimpleAnalyzer(BaseTool):
    """Простой анализатор текста"""
    
    def __init__(self):
        super().__init__()
        
        # Входной и выходной форматы
        self.input = Connector("text", "question")
        self.output = Connector("text", "answer")
        
        # Метаданные
        self.metadata.tool_name = "SimpleAnalyzer"
        self.metadata.reputation = 0.8
        self.metadata.mean_cost = 1.0
    
    def execute(self, input_data):
        """Выполнить анализ"""
        return f"Анализ вопроса '{input_data}': Это вопрос о погоде."


# ==================== Главная функция ====================

def main():
    print("\n" + "="*70)
    print("  ПРОСТОЙ ПРИМЕР GRAPHARCHITECT")
    print("="*70 + "\n")
    
    # 1. Инициализация сервисов
    print("1. Инициализация сервисов...")
    embedding_service = SimpleEmbeddingService(dimension=128)
    selector = InstrumentSelector(temperature_constant=1.0)
    finder = GraphStrategyFinder()
    orchestrator = ExecutionOrchestrator(embedding_service, selector, finder)
    print("   ✅ Готово\n")
    
    # 2. Создание инструментов
    print("2. Создание инструментов...")
    tools = [SimpleAnalyzer()]
    print(f"   ✅ Создано инструментов: {len(tools)}\n")
    
    # 3. Создание задачи
    print("3. Создание задачи...")
    task = TaskDefinition(
        description="Проанализировать вопрос",
        input_connector=Connector("text", "question"),
        output_connector=Connector("text", "answer"),
        input_data="Какая сегодня погода?"
    )
    print(f"   Вопрос: {task.input_data}")
    print("   ✅ Задача создана\n")
    
    # 4. Выполнение
    print("4. Выполнение задачи через ExecutionOrchestrator...")
    context = orchestrator.execute_task(task, tools, path_limit=1, top_k=5)
    print("   ✅ Выполнено\n")
    
    # 5. Результаты
    print("="*70)
    print("РЕЗУЛЬТАТЫ:")
    print("="*70)
    print(f"Статус:   {context.status.value}")
    print(f"Время:    {context.total_time:.4f} сек")
    print(f"Стоимость: {context.total_cost:.2f}")
    print(f"Шагов:    {context.get_total_steps()}")
    
    if context.result:
        print(f"\nРезультат: {context.result}")
    
    # 6. Детали выполнения
    if context.execution_steps:
        print("\nДетали:")
        for step in context.execution_steps:
            print(f"  Шаг {step.step_number}:")
            print(f"    Инструмент: {step.selected_tool.metadata.tool_name}")
            
            if step.selection_result:
                print(f"    Вероятность: {step.selection_result.selection_probability:.3f}")
                print(f"    Температура: {step.selection_result.temperature:.3f}")
    
    print("\n" + "="*70)
    print("  ПРИМЕР ЗАВЕРШЕН")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
