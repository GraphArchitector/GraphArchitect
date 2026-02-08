"""
Пример 1: Базовое использование NLI

Демонстрирует как использовать Natural Language Interface (NLI) для парсинга
описаний задач на естественном языке в коннекторы входа/выхода.

NLI преобразует текст вроде "Классифицировать этот текст" в структурированный формат:
    Вход:  text|question
    Выход: text|category
"""

import sys
from pathlib import Path

# Добавляем библиотеку GraphArchitect в путь
grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

from grapharchitect.services.nli.natural_language_interface import NaturalLanguageInterface
from grapharchitect.services.nli.nli_dataset_item import NLIDatasetItem
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.entities.connectors.task_representation import TaskRepresentation


class MockTool(BaseTool):
    """Простой инструмент для демонстрации."""
    
    def __init__(self, name: str, input_format: str, output_format: str):
        super().__init__()
        self.metadata.tool_name = name
        
        input_parts = input_format.split("|")
        output_parts = output_format.split("|")
        
        self.input = Connector(input_parts[0], input_parts[1])
        self.output = Connector(output_parts[0], output_parts[1])
    
    def execute(self, input_data):
        return f"[{self.metadata.tool_name}] Обработано"


def create_sample_dataset(embedding_service):
    """
    Создать примерный датасет NLI с распространенными типами задач.
    
    Каждый пример учит NLI распознавать паттерны задач.
    """
    examples = []
    
    # Пример 1: Классификация
    rep1 = TaskRepresentation()
    rep1.input_connector = Connector("text", "question")
    rep1.output_connector = Connector("text", "category")
    rep1.description = "Классификация текста"
    
    item1 = NLIDatasetItem(
        task_text="Классифицировать этот текст по категориям",
        task_embedding=embedding_service.embed_text("Классифицировать текст"),
        representation=rep1
    )
    examples.append(item1)
    
    # Пример 2: Вопросы и ответы
    rep2 = TaskRepresentation()
    rep2.input_connector = Connector("text", "question")
    rep2.output_connector = Connector("text", "answer")
    rep2.description = "Ответы на вопросы"
    
    item2 = NLIDatasetItem(
        task_text="Ответить на этот вопрос на основе контекста",
        task_embedding=embedding_service.embed_text("Ответить на вопрос"),
        representation=rep2
    )
    examples.append(item2)
    
    # Пример 3: Генерация контента
    rep3 = TaskRepresentation()
    rep3.input_connector = Connector("text", "outline")
    rep3.output_connector = Connector("text", "content")
    rep3.description = "Генерация контента"
    
    item3 = NLIDatasetItem(
        task_text="Сгенерировать статью из этого плана",
        task_embedding=embedding_service.embed_text("Сгенерировать статью"),
        representation=rep3
    )
    examples.append(item3)
    
    # Пример 4: Суммаризация
    rep4 = TaskRepresentation()
    rep4.input_connector = Connector("text", "document")
    rep4.output_connector = Connector("text", "summary")
    rep4.description = "Суммаризация текста"
    
    item4 = NLIDatasetItem(
        task_text="Суммировать этот длинный документ",
        task_embedding=embedding_service.embed_text("Суммировать документ"),
        representation=rep4
    )
    examples.append(item4)
    
    # Пример 5: Проверка качества
    rep5 = TaskRepresentation()
    rep5.input_connector = Connector("text", "content")
    rep5.output_connector = Connector("text", "validated")
    rep5.description = "Контроль качества"
    
    item5 = NLIDatasetItem(
        task_text="Проверить качество и валидировать этот контент",
        task_embedding=embedding_service.embed_text("Проверить качество"),
        representation=rep5
    )
    examples.append(item5)
    
    return examples


def create_mock_tools():
    """Создать демо-инструменты."""
    return [
        MockTool("Classifier", "text|question", "text|category"),
        MockTool("QA-System", "text|question", "text|answer"),
        MockTool("Writer", "text|outline", "text|content"),
        MockTool("Summarizer", "text|document", "text|summary"),
        MockTool("QualityChecker", "text|content", "text|validated"),
    ]


def main():
    """Основная функция примера."""
    print("=" * 70)
    print("ПРИМЕР 1: Базовое использование NLI")
    print("=" * 70)
    print()
    print("NLI преобразует естественный язык в структурированные коннекторы")
    print("форматов данных, которые используются для планирования графа инструментов.")
    print()
    
    # Шаг 1: Инициализация компонентов
    print("Шаг 1: Инициализация NLI")
    print("-" * 70)
    
    embedding_service = SimpleEmbeddingService(dimension=384)
    nli = NaturalLanguageInterface(embedding_service)
    
    print("  [OK] NLI инициализирован")
    print()
    
    # Шаг 2: Загрузка датасета
    print("Шаг 2: Загрузка обучающих примеров")
    print("-" * 70)
    
    examples = create_sample_dataset(embedding_service)
    nli.load_dataset(examples)
    
    print(f"  [OK] Загружено {len(examples)} обучающих примеров")
    print()
    
    # Создаем инструменты
    tools = create_mock_tools()
    
    # Шаг 3: Парсинг запросов на естественном языке
    print("Шаг 3: Парсинг запросов на естественном языке")
    print("-" * 70)
    print()
    
    test_queries = [
        "Классифицировать этот запрос клиента",
        "Ответить на вопрос: Что такое AI?",
        "Создать статью о машинном обучении",
        "Суммировать эту научную статью",
        "Проверить качество этого документа"
    ]
    
    for query in test_queries:
        print(f'Запрос: "{query}"')
        
        # Парсим задачу (передаем available_tools)
        parse_result = nli.parse_task(query, tools, k=3)
        
        if parse_result.success and parse_result.task_representation:
            rep = parse_result.task_representation
            print(f"  Вход:  {rep.input_connector.format}")
            print(f"  Выход: {rep.output_connector.format}")
            print()
        else:
            print(f"  [ОШИБКА] Не удалось распарсить")
            if parse_result.error_message:
                print(f"  Причина: {parse_result.error_message}")
            print()
    
    # Шаг 4: Показываем как работает NLI внутри
    print("Шаг 4: Как работает NLI внутри")
    print("-" * 70)
    print()
    
    query = "Категоризировать это текстовое сообщение"
    print(f'Пример запроса: "{query}"')
    print()
    
    # Получаем эмбеддинг
    query_embedding = embedding_service.embed_text(query)
    print(f"1. Запрос преобразован в {len(query_embedding)}-мерный вектор")
    print()
    
    # Парсим с NLI - результат содержит похожие примеры
    parse_result = nli.parse_task(query, tools, k=3)
    
    if parse_result.success and parse_result.task_representation:
        rep = parse_result.task_representation
        
        # Показываем похожие примеры, если они есть
        if parse_result.similar_examples:
            print(f"2. Найдено {len(parse_result.similar_examples)} наиболее похожих примеров:")
            for i, scored_ex in enumerate(parse_result.similar_examples[:3], 1):
                example = scored_ex.example
                score = scored_ex.score
                print(f"   {i}. \"{example.task_text}\" (сходство: {score:.3f})")
                print(f"      {example.representation.input_connector.format} -> {example.representation.output_connector.format}")
            print()
        
        print("3. Предсказанные коннекторы из лучшего совпадения:")
        print(f"   Вход:  {rep.input_connector.format}")
        print(f"   Выход: {rep.output_connector.format}")
        print()
        
        print("4. Эти коннекторы используются для поиска подходящих инструментов в графе")
        print()
    else:
        print("[ОШИБКА] Не удалось распарсить запрос")
        print()
    
    # Итог
    print("=" * 70)
    print("Итог")
    print("=" * 70)
    print()
    print()
    print("Ключевые моменты:")
    print("  - NLI использует k-NN обучение few-shot")
    print("  - Находит похожие примеры из обучающего датасета")
    print("  - Предсказывает коннекторы на основе лучших совпадений")
    print("  - Не требуется обучение модели (k-NN непараметрический)")
    print()
    print("Далее: Используйте коннекторы в GraphStrategyFinder для планирования")
    print("=" * 70)


if __name__ == "__main__":
    main()
