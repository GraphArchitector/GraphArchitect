"""
Тесты для сущностей системы GraphArchitect.

Тестируются:
- BaseTool - базовый класс инструмента
- ToolMetadata - метаданные инструмента
- Connector - формат данных (вершина графа)
- TaskDefinition - определение задачи
- ExecutionRecord - запись о выполнении
"""

import pytest
import math
from datetime import datetime
from uuid import UUID
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.tool_metadata import ToolMetadata
from grapharchitect.entities.connectors.connector import Connector, ANY_SEMANTIC
from grapharchitect.entities.task_definition import TaskDefinition
from grapharchitect.entities.execution_record import ExecutionRecord


# ==================== Фикстуры ====================

@pytest.fixture
def sample_tool():
    """Простой тестовый инструмент"""
    class TestTool(BaseTool):
        def __init__(self):
            super().__init__()
            self.input = Connector(data_format="text", semantic_format="question")
            self.output = Connector(data_format="text", semantic_format="answer")
            self.metadata.tool_name = "TestTool"
            self.metadata.reputation = 0.8
            self.metadata.mean_cost = 5.0
        
        def execute(self, input_data):
            return f"Processed: {input_data}"
    
    return TestTool()


@pytest.fixture
def task_embedding():
    """Тестовый эмбеддинг задачи"""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


# ==================== Тесты BaseTool ====================

class TestBaseTool:
    """Тесты базового класса инструмента"""
    
    def test_tool_creation(self, sample_tool):
        """Создание инструмента"""
        assert sample_tool.metadata.tool_name == "TestTool"
        assert sample_tool.input.data_format == "text"
        assert sample_tool.output.data_format == "text"
    
    def test_reputation_property(self, sample_tool):
        """Свойство репутации"""
        assert sample_tool.reputation == 0.8
        sample_tool.reputation = 0.9
        assert sample_tool.metadata.reputation == 0.9
    
    def test_mean_cost_property(self, sample_tool):
        """Свойство средней стоимости"""
        assert sample_tool.mean_cost == 5.0
        sample_tool.mean_cost = 10.0
        assert sample_tool.metadata.mean_cost == 10.0
    
    def test_get_graph_weight(self, sample_tool):
        """Расчет веса для графа (LogLoss)"""
        weight = sample_tool.get_graph_weight()
        expected_weight = -math.log(0.8)
        assert abs(weight - expected_weight) < 1e-6
    
    def test_get_graph_weight_with_low_reputation(self):
        """Вес с низкой репутацией"""
        class LowRepTool(BaseTool):
            def __init__(self):
                super().__init__()
                self.metadata.reputation = 0.1
            def execute(self, input_data):
                return input_data
        
        tool = LowRepTool()
        weight = tool.get_graph_weight()
        assert weight > 2.0  # -log(0.1) ≈ 2.3
    
    def test_get_logit_without_embedding(self, sample_tool):
        """Логит без эмбеддинга"""
        logit = sample_tool.get_logit(None)
        expected_logit = math.log(0.8)
        assert abs(logit - expected_logit) < 1e-6
    
    def test_get_logit_with_embedding(self, sample_tool, task_embedding):
        """Логит с эмбеддингом"""
        tool_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        sample_tool.metadata.capabilities_embedding = tool_embedding
        
        logit = sample_tool.get_logit(task_embedding)
        
        # Логит = косинусное сходство + log(reputation)
        # При одинаковых векторах косинусное сходство = 1.0
        expected_logit = 1.0 + math.log(0.8)
        assert abs(logit - expected_logit) < 1e-6
    
    def test_get_temperature(self, sample_tool):
        """Вычисление температуры"""
        sample_tool.metadata.training_sample_size = 100
        sample_tool.metadata.variance_estimate = 0.25
        
        temperature = sample_tool.get_temperature(constant_c=1.0)
        
        # T = C * sqrt(variance / sample_size) = 1.0 * sqrt(0.25 / 100) = 0.05
        expected_temp = 1.0 * math.sqrt(0.25 / 100)
        assert abs(temperature - expected_temp) < 1e-6
    
    def test_execute_method(self, sample_tool):
        """Выполнение инструмента"""
        result = sample_tool.execute("test input")
        assert "Processed: test input" in result
    
    def test_clone_method(self, sample_tool):
        """Клонирование инструмента"""
        clone = sample_tool.clone()
        
        assert clone.metadata.tool_name == sample_tool.metadata.tool_name
        assert clone.reputation == sample_tool.reputation
        assert clone is not sample_tool  # Разные объекты


# ==================== Тесты ToolMetadata ====================

class TestToolMetadata:
    """Тесты метаданных инструмента"""
    
    def test_default_values(self):
        """Значения по умолчанию"""
        metadata = ToolMetadata()
        
        assert metadata.reputation == 0.5
        assert metadata.mean_cost == 1.0
        assert metadata.training_sample_size == 1
        assert metadata.variance_estimate == 1.0
        assert metadata.quality_scores == []
    
    def test_custom_values(self):
        """Пользовательские значения"""
        metadata = ToolMetadata(
            tool_name="CustomTool",
            reputation=0.9,
            mean_cost=10.0
        )
        
        assert metadata.tool_name == "CustomTool"
        assert metadata.reputation == 0.9
        assert metadata.mean_cost == 10.0
    
    def test_quality_scores_history(self):
        """История оценок качества"""
        metadata = ToolMetadata()
        metadata.quality_scores = [0.7, 0.8, 0.9]
        
        assert len(metadata.quality_scores) == 3
        # Используем abs для сравнения float с погрешностью
        avg = sum(metadata.quality_scores) / len(metadata.quality_scores)
        assert abs(avg - 0.8) < 1e-6
    
    def test_capabilities_embedding(self):
        """Эмбеддинг возможностей"""
        metadata = ToolMetadata()
        metadata.capabilities_embedding = [0.1, 0.2, 0.3]
        
        assert len(metadata.capabilities_embedding) == 3


# ==================== Тесты Connector ====================

class TestConnector:
    """Тесты коннектора (формата данных)"""
    
    def test_connector_creation(self):
        """Создание коннектора"""
        connector = Connector(
            data_format="text",
            semantic_format="question"
        )
        
        assert connector.data_format == "text"
        assert connector.semantic_format == "question"
    
    def test_format_property(self):
        """Полный формат коннектора"""
        connector = Connector(
            data_format="matrix",
            semantic_format="specter"
        )
        
        assert connector.format == "matrix|specter"
    
    def test_any_semantic_format(self):
        """Коннектор с Any семантикой"""
        connector = Connector(
            data_format="text",
            semantic_format=ANY_SEMANTIC
        )
        
        assert connector.semantic_format == "*"
        assert connector.format == "text|*"
    
    def test_input_semantic_override(self):
        """Переопределение семантики через input_semantic"""
        connector = Connector(
            data_format="text",
            semantic_format=ANY_SEMANTIC,
            input_semantic="custom"
        )
        
        # При semantic_format = "*" и input_semantic задан
        assert connector.format == "text|custom"
    
    def test_get_format_static_method(self):
        """Статический метод получения формата"""
        format_str = Connector.get_format("question", "text")
        assert format_str == "text|question"
    
    def test_clone_method(self):
        """Клонирование коннектора"""
        connector = Connector(
            data_format="json",
            semantic_format="data"
        )
        
        clone = connector.clone()
        
        assert clone.data_format == connector.data_format
        assert clone.semantic_format == connector.semantic_format
        assert clone is not connector


# ==================== Тесты TaskDefinition ====================

class TestTaskDefinition:
    """Тесты определения задачи"""
    
    def test_task_creation(self):
        """Создание задачи"""
        task = TaskDefinition(
            description="Суммаризировать текст",
            input_connector=Connector("text", "document"),
            output_connector=Connector("text", "summary")
        )
        
        assert task.description == "Суммаризировать текст"
        assert isinstance(task.task_id, UUID)
    
    def test_default_values(self):
        """Значения по умолчанию"""
        task = TaskDefinition()
        
        assert task.domain == "general"
        assert task.task_embedding is None
        assert task.input_data is None
    
    def test_with_embedding(self):
        """Задача с эмбеддингом"""
        embedding = [0.1, 0.2, 0.3]
        task = TaskDefinition(
            description="Test task",
            task_embedding=embedding
        )
        
        assert task.task_embedding == embedding
    
    def test_with_input_data(self):
        """Задача с входными данными"""
        input_data = "Sample text for processing"
        task = TaskDefinition(
            input_data=input_data
        )
        
        assert task.input_data == input_data
    
    def test_created_at_timestamp(self):
        """Метка времени создания"""
        task = TaskDefinition()
        assert isinstance(task.created_at, datetime)
        assert task.created_at <= datetime.utcnow()


# ==================== Тесты ExecutionRecord ====================

class TestExecutionRecord:
    """Тесты записи о выполнении"""
    
    def test_record_creation(self):
        """Создание записи"""
        from uuid import uuid4
        
        task_id = uuid4()
        record = ExecutionRecord(
            task_id=task_id,
            execution_time=datetime.utcnow(),
            time_taken=2.5,
            cost=10.0,
            quality_score=0.85
        )
        
        assert record.task_id == task_id
        assert record.time_taken == 2.5
        assert record.cost == 10.0
        assert record.quality_score == 0.85
    
    def test_with_embedding(self):
        """Запись с эмбеддингом"""
        from uuid import uuid4
        
        embedding = [0.1, 0.2, 0.3]
        record = ExecutionRecord(
            task_id=uuid4(),
            execution_time=datetime.utcnow(),
            time_taken=1.0,
            cost=5.0,
            quality_score=0.9,
            task_embedding=embedding
        )
        
        assert record.task_embedding == embedding
    
    def test_success_flag(self):
        """Флаг успешности"""
        from uuid import uuid4
        
        record = ExecutionRecord(
            task_id=uuid4(),
            execution_time=datetime.utcnow(),
            time_taken=1.0,
            cost=5.0,
            quality_score=0.9,
            success=True
        )
        
        assert record.success is True


# ==================== Интеграционные тесты ====================

class TestIntegration:
    """Интеграционные тесты сущностей"""
    
    def test_tool_with_task(self, sample_tool):
        """Инструмент с задачей"""
        task = TaskDefinition(
            description="Process text",
            input_connector=Connector("text", "question"),
            output_connector=Connector("text", "answer"),
            input_data="What is AI?"
        )
        
        # Проверяем совместимость форматов
        assert sample_tool.input.format == task.input_connector.format
        assert sample_tool.output.format == task.output_connector.format
    
    def test_execution_record_creation_from_tool(self, sample_tool):
        """Создание записи выполнения от инструмента"""
        from uuid import uuid4
        
        task_id = uuid4()
        
        record = ExecutionRecord(
            task_id=task_id,
            execution_time=datetime.utcnow(),
            time_taken=1.5,
            cost=sample_tool.mean_cost,
            quality_score=sample_tool.reputation,
            success=True
        )
        
        assert record.cost == sample_tool.mean_cost
        assert record.quality_score == sample_tool.reputation
    
    def test_metadata_update_simulation(self):
        """Симуляция обновления метаданных"""
        metadata = ToolMetadata(reputation=0.5)
        
        # Симулируем 10 выполнений
        for i in range(10):
            quality = 0.6 + i * 0.03  # Улучшение качества
            metadata.quality_scores.append(quality)
            metadata.training_sample_size += 1
            
            # Обновляем репутацию (простое среднее)
            metadata.reputation = (
                sum(metadata.quality_scores) / len(metadata.quality_scores)
            )
        
        # После 10 выполнений репутация должна улучшиться
        assert metadata.reputation > 0.5
        assert metadata.training_sample_size == 11  # 1 начальный + 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
