"""
Пример 3: NLI + Стратегия + Выполнение

Демонстрация полного цикла:
1. NLI парсит естественный язык
2. GraphStrategyFinder находит пути
3. ExecutionOrchestrator выполняет с softmax выбором
4. Training обновляет метрики инструментов
"""

import sys
from pathlib import Path

grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

from grapharchitect.services.nli.natural_language_interface import NaturalLanguageInterface
from grapharchitect.services.nli.nli_dataset_item import NLIDatasetItem
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.execution.execution_orchestrator import ExecutionOrchestrator
from grapharchitect.services.selection.instrument_selector import InstrumentSelector
from grapharchitect.services.training.training_orchestrator import TrainingOrchestrator
from grapharchitect.services.feedback.simple_critic import SimpleCritic
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.entities.connectors.task_representation import TaskRepresentation
from grapharchitect.entities.task_definition import TaskDefinition


class DemoTool(BaseTool):
    """Демонстрационный инструмент."""
    
    def __init__(self, name: str, input_format: str, output_format: str, reputation: float = 0.8):
        super().__init__()
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
        self.metadata.training_sample_size = 10
        self.metadata.variance_estimate = 0.1
        
        # Парсим форматы
        input_parts = input_format.split("|")
        output_parts = output_format.split("|")
        
        self.input = Connector(input_parts[0], input_parts[1])
        self.output = Connector(output_parts[0], output_parts[1])
    
    def execute(self, input_data):
        """Выполнить инструмент и вернуть результат."""
        return f"[{self.metadata.tool_name}] Обработано: {str(input_data)[:50]}"


def create_nli_examples(embedding_service):
    """Создать расширенный датасет NLI."""
    examples = []
    
    # Классификация - несколько вариантов формулировок
    classification_texts = [
        "Классифицировать текст",
        "Определить категорию",
        "Категоризировать запрос",
        "Классифицировать запрос службы поддержки",
        "Определить тип запроса"
    ]
    
    for text in classification_texts:
        rep = TaskRepresentation()
        rep.input_connector = Connector("text", "question")
        rep.output_connector = Connector("text", "category")
        
        item = NLIDatasetItem(
            task_text=text,
            task_embedding=embedding_service.embed_text(text),
            representation=rep
        )
        examples.append(item)
    
    # Вопросы-ответы
    qa_texts = [
        "Ответить на вопрос",
        "Дать ответ",
        "Предоставить ответ"
    ]
    
    for text in qa_texts:
        rep = TaskRepresentation()
        rep.input_connector = Connector("text", "question")
        rep.output_connector = Connector("text", "answer")
        
        item = NLIDatasetItem(
            task_text=text,
            task_embedding=embedding_service.embed_text(text),
            representation=rep
        )
        examples.append(item)
    
    return examples


def main():
    """Запустить полный цикл."""
    print("=" * 70)
    print("ПРИМЕР 3: NLI + Стратегия + Выполнение + Обучение")
    print("=" * 70)
    print()
    
    # Инициализация всех компонентов
    print("Инициализация компонентов GraphArchitect...")
    print("-" * 70)
    
    embedding_service = SimpleEmbeddingService(dimension=384)
    nli = NaturalLanguageInterface(embedding_service)
    selector = InstrumentSelector(temperature_constant=1.0)
    strategy_finder = GraphStrategyFinder()
    training = TrainingOrchestrator(learning_rate=0.01)
    critic = SimpleCritic()
    
    # ИСПРАВЛЕНО: Позиционные аргументы, не keyword
    orchestrator = ExecutionOrchestrator(
        embedding_service,
        selector,
        strategy_finder
    )
    
    print("  [OK] Все компоненты инициализированы")
    print()
    
    # Загрузка датасета NLI
    print("Загрузка датасета NLI...")
    print("-" * 70)
    
    examples = create_nli_examples(embedding_service)
    nli.load_dataset(examples)
    
    print(f"  [OK] {len(examples)} примеров загружено")
    
    # Показываем что в датасете
    print(f"\n  Примеры в датасете:")
    for i, ex in enumerate(examples[:3], 1):
        print(f"    {i}. \"{ex.task_text}\" → {ex.representation.input_connector.format} → {ex.representation.output_connector.format}")
    print()
    
    # Создание инструментов
    print("Создание демонстрационных инструментов...")
    print("-" * 70)
    
    tools = [
        DemoTool("GPT-4-Classifier", "text|question", "text|category", 0.95),
        DemoTool("Claude-Classifier", "text|question", "text|category", 0.90),
        DemoTool("Local-Classifier", "text|question", "text|category", 0.75),
        DemoTool("GPT-4-QA", "text|question", "text|answer", 0.92),
        DemoTool("Claude-QA", "text|question", "text|answer", 0.88),
    ]
    
    # Генерируем эмбеддинги для инструментов
    for tool in tools:
        tool.metadata.capabilities_embedding = embedding_service.embed_tool_capabilities(tool)
    
    print(f"  [OK] Создано {len(tools)} инструментов")
    print()
    
    # Тестовый запрос
    query = "Классифицировать этот запрос службы поддержки"
    
    print("=" * 70)
    print(f"ОБРАБОТКА: {query}")
    print("=" * 70)
    print()
    
    # ШАГ 1: Парсинг NLI
    print("[ШАГ 1] Парсинг NLI")
    print("-" * 70)
    
    # Передаем tools в parse_task
    parse_result = nli.parse_task(query, tools, k=3)
    
    if not parse_result.success or not parse_result.task_representation:
        print("[ОШИБКА] Парсинг NLI не удался")
        if parse_result.error_message:
            print(f"  {parse_result.error_message}")
        return
    
    representation = parse_result.task_representation
    input_connector = representation.input_connector
    output_connector = representation.output_connector
    
    print(f"  Вход:  {input_connector.format}")
    print(f"  Выход: {output_connector.format}")
    print()
    
    # ШАГ 2: Создание задачи
    print("[ШАГ 2] Создание определения задачи")
    print("-" * 70)
    
    task = TaskDefinition(
        description=query,
        input_connector=input_connector,
        output_connector=output_connector,
        input_data=query
    )
    
    print(f"  Задача создана: {task.description}")
    print(f"  Входные данные: {task.input_data}")
    print()
    
    # ШАГ 3: Выполнение задачи
    print("[ШАГ 3] Выполнение задачи")
    print("-" * 70)
    
    context = orchestrator.execute_task(
        task=task,
        available_tools=tools,
        path_limit=1,
        top_k=3
    )
    
    print(f"  Статус: {context.status.value}")
    print(f"  Шагов выполнено: {context.get_total_steps()}")
    print(f"  Общее время: {context.total_time:.3f}с")
    print(f"  Общая стоимость: ${context.total_cost:.4f}")
    print(f"  Результат: {context.result}")
    print()
    
    # ШАГ 4: Детали выполнения
    print("[ШАГ 4] Детали выполнения")
    print("-" * 70)
    
    for i, step in enumerate(context.execution_steps, 1):
        selected = step.selected_tool
        selection = step.selection_result
        
        print(f"\n  Шаг выполнения {i}:")
        print(f"    Выбранный инструмент: {selected.metadata.tool_name}")
        print(f"    Вероятность:          {selection.selection_probability:.3f} (softmax)")
        print(f"    Температура:          {selection.temperature:.3f}")
        print(f"    Репутация:            {selected.metadata.reputation:.2f}")
        
        # Показываем доступную информацию о выборе
        print(f"\n    Информация о выборе:")
        print(f"      Выбран из группы инструментов")
        print(f"      Метод: Softmax с адаптивной температурой")
        print(f"      Вероятность выбора: {selection.selection_probability:.1%}")
    
    print()
    
    # ШАГ 5: Обучение
    print("[ШАГ 5] Обучение на основе выполнения")
    print("-" * 70)
    
    # Автоматическая оценка
    feedback = critic.evaluate_execution(context)
    
    print(f"  Оценка качества: {feedback.quality_score:.3f} (авто-оценка)")
    print()
    
    # Добавление в датасет обучения
    training.add_execution_to_dataset(context, [feedback])
    
    print("  [OK] Выполнение добавлено в датасет обучения")
    print()
    
    # Обучение инструментов
    print(f"  Обучение {len(context.execution_steps)} инструмента(ов)...")
    print()
    
    # Обучаем используя метод train_all_tools
    tools_to_train = [step.selected_tool for step in context.execution_steps]
    
    # Сохраняем старые репутации
    old_reputations = {tool.metadata.tool_name: tool.metadata.reputation for tool in tools_to_train}
    
    # Обучение всех инструментов на основе датасета
    training.train_all_tools(tools_to_train)
    
    # Показываем изменения
    for tool in tools_to_train:
        old_rep = old_reputations[tool.metadata.tool_name]
        new_rep = tool.metadata.reputation
        delta = new_rep - old_rep
        
        print(f"    {tool.metadata.tool_name}:")
        print(f"      Репутация: {old_rep:.3f} -> {new_rep:.3f} (дельта: {delta:+.4f})")
    
    print()
    
    # Итог
    print("=" * 70)
    print("ЦИКЛ ЗАВЕРШЕН")
    print("=" * 70)
    print()
    print("Итог:")
    print("  1. NLI распарсил естественный язык в коннекторы")
    print("  2. GraphStrategyFinder нашел пути инструментов")
    print("  3. ExecutionOrchestrator выбрал инструменты через softmax")
    print("  4. Задача выполнена успешно")
    print("  5. Обучение обновило репутацию инструмента (Policy Gradient)")
    print()
    print("Это демонстрирует полный workflow GraphArchitect!")
    print("=" * 70)


if __name__ == "__main__":
    main()
