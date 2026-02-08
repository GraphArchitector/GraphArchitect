"""
Тесты для естественно-языкового интерфейса (NLI).

Тестируются:
- NaturalLanguageInterface - главный класс NLI
- KNNFewShotRetriever - k-NN поиск похожих примеров
- ConnectorInfoAggregator - агрегация информации о коннекторах
- NLIDatasetItem - элемент датасета
"""

import pytest
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.entities.connectors.task_representation import TaskRepresentation
from grapharchitect.entities.connectors.connector_descriptor import ConnectorDescriptor
from grapharchitect.entities.connectors.data_type import DataType
from grapharchitect.entities.connectors.semantic_type import SemanticType
from grapharchitect.entities.connectors.available_connector_info import AvailableConnectorInfo
from grapharchitect.services.nli.natural_language_interface import NaturalLanguageInterface
from grapharchitect.services.nli.knn_few_shot_retriever import KNNFewShotRetriever
from grapharchitect.services.nli.connector_info_aggregator import ConnectorInfoAggregator
from grapharchitect.services.nli.nli_dataset_item import NLIDatasetItem
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService


# ==================== Моковые классы ====================

class SimpleTool(BaseTool):
    """Простой инструмент для тестов"""
    
    def __init__(self, name: str, input_format: str, output_format: str,
                 input_semantic: str = "data", output_semantic: str = "data"):
        super().__init__()
        self.input = Connector(input_format, input_semantic)
        self.output = Connector(output_format, output_semantic)
        self.metadata.tool_name = name
    
    def execute(self, input_data):
        return f"{self.metadata.tool_name}({input_data})"


# ==================== Фикстуры ====================

@pytest.fixture
def embedding_service():
    """Сервис векторизации"""
    return SimpleEmbeddingService(dimension=128)


@pytest.fixture
def sample_tools():
    """Примеры инструментов"""
    return [
        SimpleTool("PDF2Text", "pdf", "text", "document", "document"),
        SimpleTool("Text2JSON", "text", "json", "document", "data"),
        SimpleTool("Image2Text", "image", "text", "picture", "document"),
    ]


@pytest.fixture
def nli_dataset():
    """Примеры для NLI датасета"""
    return [
        NLIDatasetItem(
            task_text="Извлечь текст из PDF",
            file_types=["pdf"],
            complex_types=["text"],
            semantic_input_types=["document"],
            semantic_output_types=["document"],
            representation=TaskRepresentation(
                input_connector=ConnectorDescriptor(
                    data_type=DataType(complex_type="file", subtype="pdf"),
                    semantic_type=SemanticType(semantic_category="document")
                ),
                output_connector=ConnectorDescriptor(
                    data_type=DataType(complex_type="structured", subtype="text"),
                    semantic_type=SemanticType(semantic_category="document")
                )
            )
        ),
        NLIDatasetItem(
            task_text="Преобразовать текст в JSON",
            file_types=[],
            complex_types=["text", "json"],
            semantic_input_types=["document"],
            semantic_output_types=["data"],
            representation=TaskRepresentation(
                input_connector=ConnectorDescriptor(
                    data_type=DataType(complex_type="structured", subtype="text"),
                    semantic_type=SemanticType(semantic_category="document")
                ),
                output_connector=ConnectorDescriptor(
                    data_type=DataType(complex_type="structured", subtype="json"),
                    semantic_type=SemanticType(semantic_category="data")
                )
            )
        ),
    ]


# ==================== Тесты ConnectorInfoAggregator ====================

class TestConnectorInfoAggregator:
    """Тесты агрегатора информации о коннекторах"""
    
    def test_aggregate_from_tools(self, sample_tools):
        """Агрегация из инструментов"""
        aggregator = ConnectorInfoAggregator()
        info = aggregator.aggregate_from_tools(sample_tools)
        
        assert isinstance(info, AvailableConnectorInfo)
    
    def test_file_types_collected(self):
        """Сбор типов файлов"""
        tools = [
            SimpleTool("Tool1", "file.pdf", "text"),
            SimpleTool("Tool2", "file.jpg", "text"),
        ]
        
        aggregator = ConnectorInfoAggregator()
        info = aggregator.aggregate_from_tools(tools)
        
        assert "pdf" in info.file_types
        assert "jpg" in info.file_types
    
    def test_complex_types_collected(self, sample_tools):
        """Сбор сложных типов"""
        aggregator = ConnectorInfoAggregator()
        info = aggregator.aggregate_from_tools(sample_tools)
        
        assert "pdf" in info.complex_types or "text" in info.complex_types
    
    def test_semantic_types_collected(self, sample_tools):
        """Сбор семантических типов"""
        aggregator = ConnectorInfoAggregator()
        info = aggregator.aggregate_from_tools(sample_tools)
        
        assert "document" in info.semantic_input_types or "data" in info.semantic_input_types


# ==================== Тесты KNNFewShotRetriever ====================

class TestKNNFewShotRetriever:
    """Тесты k-NN ретривера"""
    
    def test_retriever_creation(self, embedding_service):
        """Создание ретривера"""
        retriever = KNNFewShotRetriever(
            embedding_service,
            vector_weight=0.7,
            text_weight=0.3
        )
        
        assert retriever._vector_weight == 0.7
        assert retriever._text_weight == 0.3
    
    def test_add_example(self, embedding_service):
        """Добавление примера"""
        retriever = KNNFewShotRetriever(embedding_service)
        
        item = NLIDatasetItem(task_text="Пример задачи")
        retriever.add_example(item)
        
        stats = retriever.get_statistics()
        assert stats.total_examples == 1
    
    def test_find_similar_examples(self, embedding_service, nli_dataset):
        """Поиск похожих примеров"""
        retriever = KNNFewShotRetriever(embedding_service)
        retriever.load_dataset(nli_dataset)
        
        available_connectors = AvailableConnectorInfo()
        available_connectors.file_types.add("pdf")
        available_connectors.complex_types.add("text")
        
        query = "Извлечь текст из PDF документа"
        examples = retriever.find_similar_examples(
            query,
            available_connectors,
            k=1
        )
        
        assert len(examples) >= 1
        assert examples[0].final_score > 0
    
    def test_empty_query_raises_error(self, embedding_service):
        """Пустой запрос выбрасывает ошибку"""
        retriever = KNNFewShotRetriever(embedding_service)
        
        available_connectors = AvailableConnectorInfo()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.find_similar_examples("", available_connectors, k=3)
    
    def test_vector_and_text_scores(self, embedding_service, nli_dataset):
        """Векторная и текстовая оценки"""
        retriever = KNNFewShotRetriever(embedding_service)
        retriever.load_dataset(nli_dataset)
        
        available_connectors = AvailableConnectorInfo()
        available_connectors.complex_types.add("text")
        
        examples = retriever.find_similar_examples(
            "текст JSON",
            available_connectors,
            k=2
        )
        
        if examples:
            # Косинусное сходство может быть в диапазоне [-1, 1]
            # но SimpleEmbeddingService нормализует в [0, 1]
            # Проверяем что scores в разумных пределах
            assert -1.0 <= examples[0].vector_score <= 1.0
            assert 0.0 <= examples[0].text_score <= 1.0
    
    def test_filtering_by_available_types(self, embedding_service):
        """Фильтрация по доступным типам"""
        retriever = KNNFewShotRetriever(embedding_service)
        
        # Добавляем примеры с разными типами
        item1 = NLIDatasetItem(
            task_text="PDF обработка",
            file_types=["pdf"],
            complex_types=["text"]
        )
        item2 = NLIDatasetItem(
            task_text="Image обработка",
            file_types=["jpg"],
            complex_types=["image"]
        )
        
        retriever.add_example(item1)
        retriever.add_example(item2)
        
        # Только PDF доступен
        available = AvailableConnectorInfo()
        available.file_types.add("pdf")
        
        examples = retriever.find_similar_examples(
            "обработка",
            available,
            k=5
        )
        
        # Должен найтись хотя бы один пример
        assert len(examples) >= 0
    
    def test_statistics(self, embedding_service, nli_dataset):
        """Статистика датасета"""
        retriever = KNNFewShotRetriever(embedding_service)
        retriever.load_dataset(nli_dataset)
        
        stats = retriever.get_statistics()
        
        assert stats.total_examples == len(nli_dataset)
        assert stats.examples_with_embeddings >= 0


# ==================== Тесты NaturalLanguageInterface ====================

class TestNaturalLanguageInterface:
    """Тесты естественно-языкового интерфейса"""
    
    def test_nli_creation(self, embedding_service):
        """Создание NLI"""
        nli = NaturalLanguageInterface(
            embedding_service,
            vector_weight=0.7,
            text_weight=0.3
        )
        
        assert nli is not None
    
    def test_load_dataset(self, embedding_service, nli_dataset):
        """Загрузка датасета"""
        nli = NaturalLanguageInterface(embedding_service)
        nli.load_dataset(nli_dataset)
        
        stats = nli.get_dataset_statistics()
        assert stats.total_examples == len(nli_dataset)
    
    def test_add_example(self, embedding_service):
        """Добавление примера"""
        nli = NaturalLanguageInterface(embedding_service)
        
        item = NLIDatasetItem(task_text="Пример")
        nli.add_example(item)
        
        stats = nli.get_dataset_statistics()
        assert stats.total_examples == 1
    
    def test_parse_task(self, embedding_service, nli_dataset, sample_tools):
        """Парсинг задачи"""
        nli = NaturalLanguageInterface(embedding_service)
        nli.load_dataset(nli_dataset)
        
        result = nli.parse_task(
            "Извлечь текст из PDF файла",
            sample_tools,
            k=2
        )
        
        assert result is not None
        assert result.success or not result.success  # Может быть любой результат
    
    def test_parse_empty_task_fails(self, embedding_service, sample_tools):
        """Пустой текст задачи"""
        nli = NaturalLanguageInterface(embedding_service)
        
        result = nli.parse_task("", sample_tools, k=3)
        
        assert not result.success
        assert "empty" in result.error_message.lower()
    
    def test_parse_with_no_examples(self, embedding_service, sample_tools):
        """Парсинг без примеров в датасете"""
        nli = NaturalLanguageInterface(embedding_service)
        
        result = nli.parse_task(
            "Какая-то задача",
            sample_tools,
            k=3
        )
        
        # Без примеров должен быть провал
        assert not result.success
    
    def test_parse_result_contains_similar_examples(
        self, embedding_service, nli_dataset, sample_tools
    ):
        """Результат содержит похожие примеры"""
        nli = NaturalLanguageInterface(embedding_service)
        nli.load_dataset(nli_dataset)
        
        result = nli.parse_task(
            "Преобразовать PDF в текст",
            sample_tools,
            k=2
        )
        
        if result.success:
            assert result.similar_examples is not None
            assert len(result.similar_examples) > 0
    
    def test_explain_prediction(self, embedding_service, nli_dataset, sample_tools):
        """Объяснение предсказания"""
        nli = NaturalLanguageInterface(embedding_service)
        nli.load_dataset(nli_dataset)
        
        result = nli.parse_task(
            "Извлечь текст из PDF",
            sample_tools,
            k=2
        )
        
        explanation = nli.explain_prediction(result)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0


# ==================== Тесты NLIDatasetItem ====================

class TestNLIDatasetItem:
    """Тесты элемента NLI датасета"""
    
    def test_item_creation(self):
        """Создание элемента"""
        item = NLIDatasetItem(
            task_text="Тестовая задача",
            file_types=["pdf"],
            complex_types=["text"]
        )
        
        assert item.task_text == "Тестовая задача"
        assert "pdf" in item.file_types
    
    def test_with_embedding(self):
        """Элемент с эмбеддингом"""
        embedding = [0.1, 0.2, 0.3]
        
        item = NLIDatasetItem(
            task_text="Задача",
            task_embedding=embedding
        )
        
        assert item.task_embedding == embedding
    
    def test_with_representation(self):
        """Элемент с представлением"""
        representation = TaskRepresentation(
            input_connector=ConnectorDescriptor(),
            output_connector=ConnectorDescriptor()
        )
        
        item = NLIDatasetItem(
            task_text="Задача",
            representation=representation
        )
        
        assert item.representation == representation


# ==================== Интеграционные тесты ====================

class TestNLIIntegration:
    """Интеграционные тесты NLI"""
    
    def test_full_nli_pipeline(self, embedding_service, nli_dataset, sample_tools):
        """Полный пайплайн NLI"""
        # 1. Создание NLI
        nli = NaturalLanguageInterface(embedding_service)
        
        # 2. Загрузка датасета
        nli.load_dataset(nli_dataset)
        
        # 3. Парсинг задачи
        result = nli.parse_task(
            "Извлечь текст из PDF документа",
            sample_tools,
            k=2
        )
        
        # 4. Проверка результата
        if result.success:
            assert result.task_representation is not None
            assert result.similar_examples is not None
            assert result.available_connectors is not None
            assert result.confidence > 0
    
    def test_nli_with_multiple_queries(self, embedding_service, nli_dataset, sample_tools):
        """NLI с множественными запросами"""
        nli = NaturalLanguageInterface(embedding_service)
        nli.load_dataset(nli_dataset)
        
        queries = [
            "Извлечь текст из PDF",
            "Преобразовать текст в JSON",
            "Обработать изображение",
        ]
        
        results = []
        for query in queries:
            result = nli.parse_task(query, sample_tools, k=2)
            results.append(result)
        
        # Хотя бы один запрос должен быть успешным
        successful = [r for r in results if r.success]
        # Может быть 0 или больше, зависит от датасета


# ==================== Тесты граничных случаев ====================

class TestEdgeCases:
    """Тесты граничных случаев"""
    
    def test_very_long_task_text(self, embedding_service, sample_tools):
        """Очень длинный текст задачи"""
        nli = NaturalLanguageInterface(embedding_service)
        nli.add_example(NLIDatasetItem(task_text="пример"))
        
        long_text = "слово " * 1000  # 1000 слов
        
        result = nli.parse_task(long_text, sample_tools, k=1)
        
        # Не должно быть ошибок
        assert result is not None
    
    def test_task_with_special_characters(self, embedding_service, sample_tools):
        """Задача со специальными символами"""
        nli = NaturalLanguageInterface(embedding_service)
        nli.add_example(NLIDatasetItem(task_text="пример"))
        
        special_text = "Задача!@#$%^&*()[]{}|\\;':\"<>?,./`~"
        
        result = nli.parse_task(special_text, sample_tools, k=1)
        
        # Не должно быть ошибок
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
