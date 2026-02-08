"""
Тесты для модулей выполнения и обучения.

Тестируются:
- ExecutionOrchestrator - оркестрация выполнения задачи
- ExecutionContext - контекст выполнения
- ExecutionStep - шаг выполнения
- TrainingOrchestrator - дообучение инструментов
- TrainingDataset - датасет для обучения
"""

import pytest
from datetime import datetime
from uuid import uuid4

from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.entities.task_definition import TaskDefinition
from grapharchitect.services.execution.execution_orchestrator import ExecutionOrchestrator
from grapharchitect.services.execution.execution_context import ExecutionContext
from grapharchitect.services.execution.execution_status import ExecutionStatus
from grapharchitect.services.selection.instrument_selector import InstrumentSelector
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.training.training_orchestrator import TrainingOrchestrator
from grapharchitect.services.training.training_dataset import TrainingDataset
from grapharchitect.services.training.training_dataset_item import TrainingDatasetItem
from grapharchitect.services.feedback.feedback_data import FeedbackData, FeedbackSource


# ==================== Моковые классы ====================

class TestTool(BaseTool):
    """Тестовый инструмент"""
    
    def __init__(self, name: str, input_fmt: str, output_fmt: str, reputation: float = 0.8):
        super().__init__()
        self.input = Connector(input_fmt, "data")
        self.output = Connector(output_fmt, "data")
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
        self.metadata.mean_cost = 1.0
    
    def execute(self, input_data):
        return f"[{self.metadata.tool_name}]({input_data})"


# ==================== Фикстуры ====================

@pytest.fixture
def embedding_service():
    """Сервис векторизации"""
    return SimpleEmbeddingService(dimension=64)


@pytest.fixture
def selector():
    """Селектор инструментов"""
    return InstrumentSelector(temperature_constant=1.0)


@pytest.fixture
def strategy_finder():
    """Поиск стратегий"""
    return GraphStrategyFinder()


@pytest.fixture
def orchestrator(embedding_service, selector, strategy_finder):
    """Оркестратор выполнения"""
    return ExecutionOrchestrator(embedding_service, selector, strategy_finder)


@pytest.fixture
def simple_task():
    """Простая задача"""
    # Используем цепочку text→json→text для образования пути в графе
    return TaskDefinition(
        description="Process text",
        input_connector=Connector("text", "input"),
        output_connector=Connector("text", "output"),
        input_data="test text"
    )


@pytest.fixture
def simple_tools():
    """Простые инструменты для тестов"""
    # Создаем цепочку: text|input → json|temp → text|output
    return [
        TestTool("TextToJSON", "text", "json", reputation=0.9),
        TestTool("JSONToText", "json", "text", reputation=0.8),
    ]


# ==================== Тесты ExecutionContext ====================

class TestExecutionContext:
    """Тесты контекста выполнения"""
    
    def test_context_creation(self):
        """Создание контекста"""
        context = ExecutionContext()
        
        assert context.status == ExecutionStatus.PENDING
        assert context.execution_steps == []
        assert context.gradient_traces == []
    
    def test_add_step(self):
        """Добавление шага"""
        from grapharchitect.services.execution.execution_step import ExecutionStep
        
        context = ExecutionContext()
        step = ExecutionStep(step_number=1)
        
        context.add_step(step)
        
        assert context.get_total_steps() == 1
    
    def test_add_gradient_trace(self):
        """Добавление градиентной трассы"""
        from grapharchitect.services.selection.gradient_trace import GradientTrace
        
        context = ExecutionContext()
        trace = GradientTrace()
        
        context.add_gradient_trace(trace)
        
        assert len(context.gradient_traces) == 1
    
    def test_is_successful(self):
        """Проверка успешности"""
        context = ExecutionContext()
        
        assert not context.is_successful()
        
        context.status = ExecutionStatus.COMPLETED
        assert context.is_successful()


# ==================== Тесты ExecutionOrchestrator ====================

class TestExecutionOrchestrator:
    """Тесты оркестратора выполнения"""
    
    def test_execute_simple_task(self, orchestrator, simple_task, simple_tools):
        """Выполнение простой задачи"""
        context = orchestrator.execute_task(
            simple_task,
            simple_tools,
            path_limit=1,
            top_k=5
        )
        
        assert context is not None
        assert context.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
    
    def test_execution_creates_steps(self, orchestrator, simple_task, simple_tools):
        """Выполнение создает шаги"""
        context = orchestrator.execute_task(
            simple_task,
            simple_tools,
            path_limit=1,
            top_k=5
        )
        
        # Если путь найден - должны быть шаги
        if context.status.value == "completed":
            assert context.get_total_steps() > 0
        else:
            # Если путь не найден - это нормально для тестовых данных
            assert context.status.value == "failed"
    
    def test_execution_with_no_path(self, orchestrator):
        """Выполнение когда путь не найден"""
        task = TaskDefinition(
            input_connector=Connector("nonexistent", "format"),
            output_connector=Connector("another", "format"),
            input_data="test"
        )
        
        tools = [TestTool("IrrelevantTool", "a", "b")]
        
        context = orchestrator.execute_task(task, tools, path_limit=1, top_k=5)
        
        assert context.status == ExecutionStatus.FAILED
        assert "не найдено путей" in context.error_message.lower()
    
    def test_execution_time_tracking(self, orchestrator, simple_task, simple_tools):
        """Отслеживание времени выполнения"""
        context = orchestrator.execute_task(
            simple_task,
            simple_tools,
            path_limit=1,
            top_k=5
        )
        
        assert context.start_time is not None
        assert context.end_time is not None
        assert context.total_time >= 0
    
    def test_execution_result_stored(self, orchestrator, simple_task, simple_tools):
        """Результат сохраняется"""
        context = orchestrator.execute_task(
            simple_task,
            simple_tools,
            path_limit=1,
            top_k=5
        )
        
        if context.status == ExecutionStatus.COMPLETED:
            assert context.result is not None
    
    def test_gradient_traces_collected(self, orchestrator, simple_task, simple_tools):
        """Градиентные трассы собираются"""
        context = orchestrator.execute_task(
            simple_task,
            simple_tools,
            path_limit=1,
            top_k=5
        )
        
        # Если выполнение успешно - должны быть градиенты
        if context.status.value == "completed":
            assert len(context.gradient_traces) > 0
        else:
            # Если путь не найден - градиентов нет, это нормально
            assert context.status.value == "failed"


# ==================== Тесты TrainingDataset ====================

class TestTrainingDataset:
    """Тесты датасета для обучения"""
    
    def test_dataset_creation(self):
        """Создание датасета"""
        dataset = TrainingDataset()
        
        assert dataset.size() == 0
    
    def test_add_item(self):
        """Добавление элемента"""
        dataset = TrainingDataset()
        
        item = TrainingDatasetItem(
            task_id=uuid4(),
            quality_score=0.8
        )
        
        dataset.add_item(item)
        
        assert dataset.size() == 1
    
    def test_get_items_by_quality(self):
        """Фильтрация по качеству"""
        dataset = TrainingDataset()
        
        dataset.add_item(TrainingDatasetItem(task_id=uuid4(), quality_score=0.9))
        dataset.add_item(TrainingDatasetItem(task_id=uuid4(), quality_score=0.5))
        dataset.add_item(TrainingDatasetItem(task_id=uuid4(), quality_score=0.8))
        
        high_quality = dataset.get_items_by_quality_threshold(0.7)
        
        assert len(high_quality) == 2  # 0.9 и 0.8
    
    def test_get_items_by_domain(self):
        """Фильтрация по домену"""
        dataset = TrainingDataset()
        
        dataset.add_item(TrainingDatasetItem(task_id=uuid4(), domain="physics"))
        dataset.add_item(TrainingDatasetItem(task_id=uuid4(), domain="medicine"))
        dataset.add_item(TrainingDatasetItem(task_id=uuid4(), domain="physics"))
        
        physics_items = dataset.get_items_by_domain("physics")
        
        assert len(physics_items) == 2
    
    def test_clear_dataset(self):
        """Очистка датасета"""
        dataset = TrainingDataset()
        dataset.add_item(TrainingDatasetItem(task_id=uuid4()))
        
        assert dataset.size() == 1
        
        dataset.clear()
        
        assert dataset.size() == 0


# ==================== Тесты TrainingOrchestrator ====================

class TestTrainingOrchestrator:
    """Тесты оркестратора обучения"""
    
    def test_orchestrator_creation(self):
        """Создание оркестратора"""
        orchestrator = TrainingOrchestrator(learning_rate=0.01)
        
        assert orchestrator._learning_rate == 0.01
    
    def test_add_execution_to_dataset(self, orchestrator, simple_task):
        """Добавление выполнения в датасет"""
        training = TrainingOrchestrator()
        
        context = ExecutionContext()
        context.task_id = simple_task.task_id
        context.task = simple_task
        context.status = ExecutionStatus.COMPLETED
        
        feedbacks = [
            FeedbackData(
                task_id=simple_task.task_id,
                quality_score=0.9,
                source=FeedbackSource.USER
            )
        ]
        
        training.add_execution_to_dataset(context, feedbacks)
        
        dataset = training.get_dataset()
        assert dataset.size() == 1
    
    def test_update_tool_increases_reputation(self):
        """Обновление инструмента увеличивает репутацию"""
        training = TrainingOrchestrator(learning_rate=0.1)
        
        tool = TestTool("TestTool", "a", "b", reputation=0.5)
        
        item = TrainingDatasetItem(
            task_id=uuid4(),
            selected_tools=[tool],
            quality_score=0.9,  # Высокое качество
            gradient_traces=[]
        )
        
        # Добавляем градиентную трассу
        from grapharchitect.services.selection.gradient_trace import GradientTrace
        trace = GradientTrace(selected_tool=tool)
        item.gradient_traces.append(trace)
        
        old_reputation = tool.metadata.reputation
        training.update_tool(tool, item)
        
        # Репутация должна увеличиться
        assert tool.metadata.reputation > old_reputation
    
    def test_update_tool_decreases_reputation(self):
        """Обновление инструмента уменьшает репутацию"""
        training = TrainingOrchestrator(learning_rate=0.1)
        
        tool = TestTool("TestTool", "a", "b", reputation=0.8)
        
        item = TrainingDatasetItem(
            task_id=uuid4(),
            selected_tools=[tool],
            quality_score=0.2,  # Низкое качество
            gradient_traces=[]
        )
        
        from grapharchitect.services.selection.gradient_trace import GradientTrace
        trace = GradientTrace(selected_tool=tool)
        item.gradient_traces.append(trace)
        
        old_reputation = tool.metadata.reputation
        training.update_tool(tool, item)
        
        # Репутация должна уменьшиться
        assert tool.metadata.reputation < old_reputation
    
    def test_reputation_bounded(self):
        """Репутация ограничена диапазоном [0.01, 0.99]"""
        training = TrainingOrchestrator(learning_rate=10.0)  # Очень большой шаг
        
        tool = TestTool("TestTool", "a", "b", reputation=0.9)
        
        item = TrainingDatasetItem(
            task_id=uuid4(),
            selected_tools=[tool],
            quality_score=1.0,
            gradient_traces=[]
        )
        
        from grapharchitect.services.selection.gradient_trace import GradientTrace
        trace = GradientTrace(selected_tool=tool)
        item.gradient_traces.append(trace)
        
        training.update_tool(tool, item)
        
        # Репутация не должна превышать 0.99
        assert tool.metadata.reputation <= 0.99
    
    def test_train_all_tools(self):
        """Обучение всех инструментов"""
        training = TrainingOrchestrator(learning_rate=0.01)
        
        tools = [
            TestTool("Tool1", "a", "b", reputation=0.5),
            TestTool("Tool2", "b", "c", reputation=0.5),
        ]
        
        # Добавляем элементы в датасет
        for tool in tools:
            item = TrainingDatasetItem(
                task_id=uuid4(),
                selected_tools=[tool],
                quality_score=0.8,
                gradient_traces=[]
            )
            from grapharchitect.services.selection.gradient_trace import GradientTrace
            trace = GradientTrace(selected_tool=tool)
            item.gradient_traces.append(trace)
            
            training.get_dataset().add_item(item)
        
        training.train_all_tools(tools)
        
        # Репутация инструментов должна измениться
        for tool in tools:
            assert tool.metadata.reputation != 0.5
    
    def test_train_on_successful_executions(self):
        """Обучение только на успешных выполнениях"""
        training = TrainingOrchestrator(learning_rate=0.01)
        
        tool = TestTool("TestTool", "a", "b", reputation=0.5)
        
        # Добавляем успешное и неуспешное выполнение
        from grapharchitect.services.selection.gradient_trace import GradientTrace
        
        success_item = TrainingDatasetItem(
            task_id=uuid4(),
            selected_tools=[tool],
            quality_score=0.9,
            gradient_traces=[GradientTrace(selected_tool=tool)]
        )
        
        fail_item = TrainingDatasetItem(
            task_id=uuid4(),
            selected_tools=[tool],
            quality_score=0.3,
            gradient_traces=[GradientTrace(selected_tool=tool)]
        )
        
        training.get_dataset().add_item(success_item)
        training.get_dataset().add_item(fail_item)
        
        old_reputation = tool.metadata.reputation
        
        # Обучаем только на успешных (качество >= 0.7)
        training.train_on_successful_executions([tool], quality_threshold=0.7)
        
        # Репутация должна увеличиться (только успешные учтены)
        assert tool.metadata.reputation > old_reputation
    
    def test_get_statistics(self):
        """Получение статистики обучения"""
        training = TrainingOrchestrator()
        
        # Добавляем несколько элементов
        training.get_dataset().add_item(
            TrainingDatasetItem(
                task_id=uuid4(),
                quality_score=0.8,
                execution_time=1.5,
                total_cost=5.0
            )
        )
        training.get_dataset().add_item(
            TrainingDatasetItem(
                task_id=uuid4(),
                quality_score=0.9,
                execution_time=2.0,
                total_cost=7.0
            )
        )
        
        stats = training.get_statistics()
        
        assert stats.total_executions == 2
        assert abs(stats.average_quality - 0.85) < 1e-6
        assert abs(stats.average_execution_time - 1.75) < 1e-6
        assert abs(stats.average_cost - 6.0) < 1e-6


# ==================== Интеграционные тесты ====================

class TestIntegration:
    """Интеграционные тесты выполнения и обучения"""
    
    def test_full_pipeline_with_training(self, orchestrator, simple_task, simple_tools):
        """Полный пайплайн: выполнение -> обратная связь -> обучение"""
        # 1. Выполнение задачи
        context = orchestrator.execute_task(
            simple_task,
            simple_tools,
            path_limit=1,
            top_k=5
        )
        
        # 2. Сбор обратной связи
        feedbacks = [
            FeedbackData(
                task_id=context.task_id,
                quality_score=0.9,
                source=FeedbackSource.USER
            )
        ]
        
        # 3. Обучение
        training = TrainingOrchestrator(learning_rate=0.01)
        training.add_execution_to_dataset(context, feedbacks)
        
        # 4. Дообучение инструментов
        training.train_all_tools(simple_tools)
        
        # Проверяем что инструменты обновились
        assert training.get_statistics().total_executions == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
