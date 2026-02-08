"""
Тесты для сервисов системы GraphArchitect.

Тестируются:
- GraphBuilder - построение графа из инструментов
- GraphStrategyFinder - поиск стратегий
- ToolEdge - ребро графа с группой инструментов
- ToolExpander - расширение Any->Any инструментов
- EmbeddingService - векторизация
- FeedbackCollector - сбор обратной связи
- SimpleCritic - автоматическая оценка
"""

import pytest
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector, ANY_SEMANTIC
from grapharchitect.entities.task_definition import TaskDefinition
from grapharchitect.services.graph_builder import GraphBuilder
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.tool_edge import ToolEdge
from grapharchitect.services.tool_expander import ToolExpander
from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.feedback.feedback_collector import FeedbackCollector
from grapharchitect.services.feedback.feedback_data import FeedbackData, FeedbackSource
from grapharchitect.services.feedback.simple_critic import SimpleCritic
from grapharchitect.services.execution.execution_context import ExecutionContext
from grapharchitect.services.execution.execution_status import ExecutionStatus


# ==================== Моковые классы ====================

class SimpleTool(BaseTool):
    """Простой инструмент для тестов"""
    
    def __init__(self, name: str, input_format: str, output_format: str,
                 input_semantic: str = "data", output_semantic: str = "data",
                 reputation: float = 0.8):
        super().__init__()
        self.input = Connector(input_format, input_semantic)
        self.output = Connector(output_format, output_semantic)
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
    
    def execute(self, input_data):
        return f"{self.metadata.tool_name}({input_data})"


# ==================== Фикстуры ====================

@pytest.fixture
def basic_tools():
    """Базовый набор инструментов для простых путей"""
    return [
        SimpleTool("PDF2Text", "pdf", "text", "document", "document"),
        SimpleTool("Text2JSON", "text", "json", "document", "data"),
        SimpleTool("JSON2XML", "json", "xml", "data", "data"),
    ]


@pytest.fixture
def tools_with_alternatives():
    """Инструменты с альтернативными путями"""
    return [
        # Прямой путь
        SimpleTool("PDF2Text_Direct", "pdf", "text", "doc", "doc", reputation=0.5),
        
        # Альтернативные пути
        SimpleTool("PDF2Image", "pdf", "image", "doc", "doc", reputation=0.8),
        SimpleTool("Image2Text", "image", "text", "doc", "doc", reputation=0.8),
        
        SimpleTool("PDF2HTML", "pdf", "html", "doc", "doc", reputation=0.9),
        SimpleTool("HTML2Text", "html", "text", "doc", "doc", reputation=0.9),
    ]


@pytest.fixture
def any_tools():
    """Инструменты с Any семантикой"""
    return [
        SimpleTool("Universal2JSON", "text", "json", ANY_SEMANTIC, "data"),
        SimpleTool("PDF2Text", "pdf", "text", "document", "document"),
        SimpleTool("XML2Text", "xml", "text", "document", "document"),
    ]


# ==================== Тесты GraphBuilder ====================

class TestGraphBuilder:
    """Тесты построителя графа"""
    
    def test_build_simple_graph(self, basic_tools):
        """Построение простого графа"""
        builder = GraphBuilder()
        graph = builder.build(basic_tools)
        
        assert graph.v >= 4  # Минимум 4 уникальных формата
        assert graph.e == 3  # 3 ребра (по одному на каждый инструмент)
    
    def test_vertex_mapping(self, basic_tools):
        """Маппинг формат -> ID вершины"""
        builder = GraphBuilder()
        graph = builder.build(basic_tools)
        
        pdf_id = builder.get_node_id("pdf|document")
        text_id = builder.get_node_id("text|document")
        
        assert pdf_id is not None
        assert text_id is not None
        assert pdf_id != text_id
    
    def test_get_format_by_id(self, basic_tools):
        """Получение формата по ID"""
        builder = GraphBuilder()
        graph = builder.build(basic_tools)
        
        pdf_id = builder.get_node_id("pdf|document")
        format_str = builder.get_format_by_id(pdf_id)
        
        assert format_str == "pdf|document"
    
    def test_multiple_tools_same_edge(self):
        """Несколько инструментов на одном ребре"""
        tools = [
            SimpleTool("Tool_A", "pdf", "text", reputation=0.9),
            SimpleTool("Tool_B", "pdf", "text", reputation=0.7),
            SimpleTool("Tool_C", "pdf", "text", reputation=0.5),
        ]
        
        builder = GraphBuilder()
        graph = builder.build(tools)
        
        # Все инструменты должны быть на одном ребре
        assert graph.e == 1
        
        # Найдем это ребро
        pdf_id = builder.get_node_id("pdf|data")
        edges = graph.adj(pdf_id)
        
        assert len(edges) == 1
        assert len(edges[0].tools) == 3


# ==================== Тесты ToolEdge ====================

class TestToolEdge:
    """Тесты ребра с группой инструментов"""
    
    def test_edge_creation(self):
        """Создание ребра"""
        tool = SimpleTool("TestTool", "a", "b")
        edge = ToolEdge(0, 1, tool)
        
        assert edge.start_v == 0
        assert edge.end_v == 1
        assert len(edge.tools) == 1
    
    def test_add_tool(self):
        """Добавление инструмента"""
        edge = ToolEdge(0, 1)
        
        tool1 = SimpleTool("Tool1", "a", "b", reputation=0.8)
        tool2 = SimpleTool("Tool2", "a", "b", reputation=0.6)
        
        edge.add_tool(tool1)
        edge.add_tool(tool2)
        
        assert len(edge.tools) == 2
    
    def test_weight_calculation(self):
        """Вес ребра = среднее по всем инструментам"""
        edge = ToolEdge(0, 1)
        
        tool1 = SimpleTool("Tool1", "a", "b", reputation=0.9)  # вес ≈ 0.105
        tool2 = SimpleTool("Tool2", "a", "b", reputation=0.5)  # вес ≈ 0.693
        
        edge.add_tool(tool1)
        edge.add_tool(tool2)
        
        # Вес должен быть средним
        expected_weight = (tool1.get_graph_weight() + tool2.get_graph_weight()) / 2
        assert abs(edge.w - expected_weight) < 1e-6
    
    def test_get_group_logits(self):
        """Получение логитов группы"""
        edge = ToolEdge(0, 1)
        
        tool1 = SimpleTool("Tool1", "a", "b", reputation=0.9)
        tool2 = SimpleTool("Tool2", "a", "b", reputation=0.5)
        
        edge.add_tool(tool1)
        edge.add_tool(tool2)
        
        logits = edge.get_group_logits(None)
        
        assert len(logits) == 2
        assert logits[tool1] > logits[tool2]  # Выше репутация -> больше логит


# ==================== Тесты ToolExpander ====================

class TestToolExpander:
    """Тесты расширителя Any->Any инструментов"""
    
    def test_no_expansion_needed(self, basic_tools):
        """Расширение не требуется"""
        expander = ToolExpander()
        expanded = expander.expand(basic_tools, "document", "data")
        
        # Количество инструментов не должно измениться
        assert len(expanded) == len(basic_tools)
    
# ==================== Тесты GraphStrategyFinder ====================

class TestGraphStrategyFinder:
    """Тесты поиска стратегий"""
    
    def test_find_direct_path(self, basic_tools):
        """Поиск прямого пути"""
        finder = GraphStrategyFinder()
        
        strategies = finder.find_strategies(
            basic_tools,
            "pdf|document",
            "text|document",
            limit=1
        )
        
        assert len(strategies) == 1
        assert len(strategies[0]) == 1  # Один шаг
        assert strategies[0][0].metadata.tool_name == "PDF2Text"
    
    def test_find_multi_step_path(self, basic_tools):
        """Поиск многошагового пути"""
        finder = GraphStrategyFinder()
        
        strategies = finder.find_strategies(
            basic_tools,
            "pdf|document",
            "json|data",
            limit=1
        )
        
        assert len(strategies) == 1
        assert len(strategies[0]) == 2  # Два шага: PDF->Text->JSON
    
    def test_no_path_exists(self):
        """Путь не существует"""
        tools = [
            SimpleTool("Tool1", "a", "b"),
            SimpleTool("Tool2", "c", "d"),
        ]
        
        finder = GraphStrategyFinder()
        strategies = finder.find_strategies(tools, "a|data", "d|data", limit=1)
        
        assert strategies == []
    
    def test_algorithm_selection_dijkstra(self, basic_tools):
        """Использование алгоритма Dijkstra"""
        finder = GraphStrategyFinder()
        
        strategies = finder.find_strategies(
            basic_tools,
            "pdf|document",
            "text|document",
            limit=1,
            algorithm=PathfindingAlgorithm.DIJKSTRA
        )
        
        assert len(strategies) >= 1
    
    def test_algorithm_selection_yen(self, tools_with_alternatives):
        """Использование алгоритма Yen для множественных путей"""
        finder = GraphStrategyFinder()
        
        strategies = finder.find_strategies(
            tools_with_alternatives,
            "pdf|doc",
            "text|doc",
            limit=3,
            algorithm=PathfindingAlgorithm.YEN
        )
        
        # Должно быть найдено несколько путей
        assert len(strategies) >= 1


# ==================== Тесты EmbeddingService ====================

class TestEmbeddingService:
    """Тесты сервиса векторизации"""
    
    def test_embed_text(self):
        """Векторизация текста"""
        service = SimpleEmbeddingService(dimension=128)
        
        text = "Пример текста для векторизации"
        embedding = service.embed_text(text)
        
        assert len(embedding) == 128
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_empty_text(self):
        """Векторизация пустого текста"""
        service = SimpleEmbeddingService(dimension=64)
        
        embedding = service.embed_text("")
        
        assert len(embedding) == 64
        assert all(x == 0.0 for x in embedding)
    
    def test_embed_tool_capabilities(self):
        """Векторизация возможностей инструмента"""
        service = SimpleEmbeddingService(dimension=128)
        tool = SimpleTool("TestTool", "pdf", "text")
        
        embedding = service.embed_tool_capabilities(tool)
        
        assert len(embedding) == 128
    
    def test_compute_similarity(self):
        """Вычисление схожести"""
        service = SimpleEmbeddingService()
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        # Идентичные векторы
        sim_identical = service.compute_similarity(vec1, vec2)
        assert abs(sim_identical - 1.0) < 0.1  # Близко к 1
        
        # Ортогональные векторы
        sim_orthogonal = service.compute_similarity(vec1, vec3)
        assert abs(sim_orthogonal - 0.5) < 0.1  # Близко к 0.5 (нормализовано)
    
    def test_consistent_embeddings(self):
        """Консистентность эмбеддингов"""
        service = SimpleEmbeddingService()
        
        text = "Тестовый текст"
        emb1 = service.embed_text(text)
        emb2 = service.embed_text(text)
        
        # Одинаковый текст -> одинаковый эмбеддинг
        assert emb1 == emb2


# ==================== Тесты Feedback ====================

class TestFeedbackCollector:
    """Тесты сборщика обратной связи"""
    
    def test_add_feedback(self):
        """Добавление обратной связи"""
        from uuid import uuid4
        
        collector = FeedbackCollector()
        task_id = uuid4()
        
        feedback = FeedbackData(
            task_id=task_id,
            source=FeedbackSource.USER,
            quality_score=0.9
        )
        
        collector.add_feedback(feedback)
        feedbacks = collector.get_feedbacks(task_id)
        
        assert len(feedbacks) == 1
        assert feedbacks[0].quality_score == 0.9
    
    def test_multiple_feedbacks(self):
        """Несколько отзывов для одной задачи"""
        from uuid import uuid4
        
        collector = FeedbackCollector()
        task_id = uuid4()
        
        collector.add_feedback(FeedbackData(task_id=task_id, quality_score=0.8))
        collector.add_feedback(FeedbackData(task_id=task_id, quality_score=0.9))
        
        feedbacks = collector.get_feedbacks(task_id)
        assert len(feedbacks) == 2
    
    def test_average_quality(self):
        """Средняя оценка качества"""
        from uuid import uuid4
        
        collector = FeedbackCollector()
        task_id = uuid4()
        
        collector.add_feedback(FeedbackData(task_id=task_id, quality_score=0.6))
        collector.add_feedback(FeedbackData(task_id=task_id, quality_score=0.8))
        collector.add_feedback(FeedbackData(task_id=task_id, quality_score=1.0))
        
        avg = collector.get_average_quality(task_id)
        assert abs(avg - 0.8) < 1e-6


class TestSimpleCritic:
    """Тесты автоматического критика"""
    
    def test_evaluate_successful_execution(self):
        """Оценка успешного выполнения"""
        critic = SimpleCritic()
        
        context = ExecutionContext()
        context.status = ExecutionStatus.COMPLETED
        context.result = "Some result"
        
        feedback = critic.evaluate_execution(context)
        
        assert feedback.success is True
        assert feedback.quality_score > 0.0
    
    def test_evaluate_failed_execution(self):
        """Оценка неудачного выполнения"""
        critic = SimpleCritic()
        
        context = ExecutionContext()
        context.status = ExecutionStatus.FAILED
        
        feedback = critic.evaluate_execution(context)
        
        assert feedback.success is False
        assert feedback.quality_score == 0.0
    
    def test_detect_errors_with_none_result(self):
        """Обнаружение ошибки с None результатом"""
        critic = SimpleCritic()
        
        errors = critic.detect_errors(None)
        
        assert len(errors) > 0
        assert "None" in errors[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
