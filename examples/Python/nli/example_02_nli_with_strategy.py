"""
Пример 2: NLI + Поиск стратегий

Демонстрирует полный workflow:
1. NLI парсит естественный язык в коннекторы
2. GraphStrategyFinder использует коннекторы для поиска путей инструментов
3. Показывает как они работают вместе
"""

import sys
from pathlib import Path

grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

from grapharchitect.services.nli.natural_language_interface import NaturalLanguageInterface
from grapharchitect.services.nli.nli_dataset_item import NLIDatasetItem
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.entities.connectors.task_representation import TaskRepresentation


class SimpleTool(BaseTool):
    """Простой инструмент для демонстрации."""
    
    def __init__(self, name: str, input_format: str, output_format: str, reputation: float = 0.8):
        super().__init__()
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
        
        # Парсим формат "data|semantic"
        input_parts = input_format.split("|")
        output_parts = output_format.split("|")
        
        self.input = Connector(input_parts[0], input_parts[1])
        self.output = Connector(output_parts[0], output_parts[1])
    
    def execute(self, input_data):
        return f"[{self.metadata.tool_name}] Обработано: {input_data}"


def create_nli_dataset(embedding_service):
    """Создать обучающие примеры NLI."""
    examples = []
    
    tasks = [
        ("Классифицировать текст", "text|question", "text|category", "Классификация"),
        ("Ответить на вопрос", "text|question", "text|answer", "Ответы"),
        ("Сгенерировать контент", "text|outline", "text|content", "Генерация"),
        ("Суммировать документ", "text|document", "text|summary", "Суммаризация"),
        ("Проверить качество", "text|content", "text|validated", "Контроль качества"),
    ]
    
    for text, input_fmt, output_fmt, desc in tasks:
        rep = TaskRepresentation()
        
        input_parts = input_fmt.split("|")
        output_parts = output_fmt.split("|")
        
        rep.input_connector = Connector(input_parts[0], input_parts[1])
        rep.output_connector = Connector(output_parts[0], output_parts[1])
        rep.description = desc
        
        item = NLIDatasetItem(
            task_text=text,
            task_embedding=embedding_service.embed_text(text),
            representation=rep
        )
        examples.append(item)
    
    return examples


def create_mock_tools():
    """Создать набор инструментов покрывающих разные преобразования форматов."""
    tools = [
        SimpleTool("Classifier-A", "text|question", "text|category", 0.90),
        SimpleTool("Classifier-B", "text|question", "text|category", 0.85),
        SimpleTool("QA-System", "text|question", "text|answer", 0.88),
        SimpleTool("Content-Writer", "text|outline", "text|content", 0.82),
        SimpleTool("Summarizer", "text|document", "text|summary", 0.86),
        SimpleTool("Quality-Checker", "text|content", "text|validated", 0.84),
    ]
    
    return tools


def main():
    """Основная функция примера."""
    print("=" * 70)
    print("ПРИМЕР 2: NLI + Поиск стратегий")
    print("=" * 70)
    print()
    
    # Инициализация компонентов
    print("Инициализация компонентов...")
    print("-" * 70)
    
    embedding_service = SimpleEmbeddingService(dimension=384)
    nli = NaturalLanguageInterface(embedding_service)
    strategy_finder = GraphStrategyFinder()
    
    print("  [OK] NLI инициализирован")
    print("  [OK] Поиск стратегий инициализирован")
    print()
    
    # Загрузка датасета NLI
    print("Загрузка обучающих примеров NLI...")
    print("-" * 70)
    
    examples = create_nli_dataset(embedding_service)
    nli.load_dataset(examples)
    
    print(f"  [OK] Загружено {len(examples)} примеров")
    print()
    
    # Создание инструментов
    print("Создание демо-инструментов...")
    print("-" * 70)
    
    tools = create_mock_tools()
    
    for tool in tools:
        print(f"  {tool.metadata.tool_name:20} {tool.input.format} -> {tool.output.format}")
    
    print()
    
    # Тестовые запросы
    test_queries = [
        "Классифицировать этот тикет поддержки клиента",
        "Ответить: Что такое машинное обучение?",
        "Создать подробную статью об AI"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print("=" * 70)
        print(f"ТЕСТ {i}: {query}")
        print("=" * 70)
        print()
        
        # Шаг 1: NLI парсит запрос
        print("Шаг 1: Парсинг NLI")
        print("-" * 70)
        
        # Передаем tools в parse_task
        parse_result = nli.parse_task(query, tools, k=3)
        
        if not parse_result.success or not parse_result.task_representation:
            print(f"[ОШИБКА] Не удалось распарсить запрос")
            if parse_result.error_message:
                print(f"  {parse_result.error_message}")
            print()
            continue
        
        representation = parse_result.task_representation
        input_connector = representation.input_connector
        output_connector = representation.output_connector
        
        print(f"  Запрос:  \"{query}\"")
        print(f"  Вход:  {input_connector.format}")
        print(f"  Выход: {output_connector.format}")
        print()
        
        # Шаг 2: Поиск стратегий в графе инструментов
        print("Шаг 2: Поиск стратегий инструментов")
        print("-" * 70)
        
        strategies = strategy_finder.find_strategies(
            tools=tools,
            start_format=input_connector.format,
            end_format=output_connector.format,
            algorithm=PathfindingAlgorithm.DIJKSTRA,
            limit=3
        )
        
        if strategies:
            print(f"  [OK] Найдено {len(strategies)} стратегий")
            print()
            
            for j, strategy in enumerate(strategies, 1):
                tool_names = [t.metadata.tool_name for t in strategy]
                tool_chain = " -> ".join(tool_names)
                
                # Вычисляем общую стоимость (сумма log-loss весов)
                total_weight = sum(-1.0 * t.get_graph_weight() for t in strategy)
                
                print(f"  Стратегия {j}: {tool_chain}")
                print(f"    Инструментов: {len(strategy)}")
                print(f"    Вес: {total_weight:.3f}")
                print()
        else:
            print(f"  [ПРЕДУПРЕЖДЕНИЕ] Стратегии не найдены")
            print(f"  Нет пути от {input_connector.format} до {output_connector.format}")
            print()
        
        print()
    
    # Итог
    print("=" * 70)
    print("Полный workflow")
    print("=" * 70)
    print()
    print("1. Пользователь предоставляет запрос на естественном языке")
    print("   \"Классифицировать этот тикет поддержки клиента\"")
    print()
    print("2. NLI парсит в коннекторы")
    print("   text|question -> text|category")
    print()
    print("3. GraphStrategyFinder находит пути инструментов")
    print("   [Classifier-A] или [Classifier-B]")
    print()
    print("4. ExecutionOrchestrator выбирает и выполняет")
    print("   (см. example_03 для выполнения)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
