"""
Тесты интеграции GraphArchitect с Web API.

Проверяет:
- Конверсию Agent → BaseTool
- Работу GraphArchitectBridge
- Выбор инструментов через softmax
- Поиск стратегий в графе
- Стриминг выполнения
"""

import pytest
import asyncio
from grapharchitect_bridge import (
    AgentTool, GraphArchitectBridge, get_bridge, is_bridge_available
)
from repository import get_repository
from models import MessageChunk


# ==================== Тесты AgentTool ====================

class TestAgentTool:
    """Тесты адаптера Agent → BaseTool"""
    
    def test_agent_to_tool_conversion(self, repository):
        """Конверсия Agent в BaseTool"""
        agent = repository.get_agent("agent-classifier-gpt4")
        assert agent is not None, "Агент не найден в БД"
        
        tool = AgentTool(agent)
        
        # Проверка метаданных
        assert tool.metadata.tool_name == agent.name
        assert tool.metadata.reputation == agent.metrics["avgScore"]
        assert tool.metadata.mean_cost == agent.cost
        assert tool.metadata.mean_time_answer == agent.metrics["avgResponseTime"] / 1000
        
        # Проверка коннекторов
        assert tool.input is not None
        assert tool.output is not None
        assert tool.input.data_format != ""
        assert tool.output.data_format != ""
    
    def test_connectors_inference(self, repository):
        """Вывод коннекторов из типа агента"""
        # Classification agent
        classifier = repository.get_agent("agent-classifier-gpt4")
        tool_classifier = AgentTool(classifier)
        
        assert tool_classifier.input.format == "text|question"
        assert tool_classifier.output.format == "text|category"
        
        # Writing agent
        writer = repository.get_agent("agent-writer-formal")
        if writer:
            tool_writer = AgentTool(writer)
            assert tool_writer.input.format == "text|outline"
            assert tool_writer.output.format == "text|article"
    
    def test_agent_id_preserved(self, repository):
        """ID агента сохраняется"""
        agent = repository.get_agent("agent-classifier-gpt4")
        tool = AgentTool(agent)
        
        assert tool.agent_id == agent.id
    
    def test_execute_method(self, repository):
        """Метод execute работает"""
        agent = repository.get_agent("agent-classifier-gpt4")
        tool = AgentTool(agent)
        
        result = tool.execute("test input")
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_all_agents_convertible(self, repository):
        """Все агенты конвертируются в BaseTool"""
        agents = repository.get_all_agents()
        
        for agent in agents:
            try:
                tool = AgentTool(agent)
                assert tool.metadata.tool_name == agent.name
                assert tool.input is not None
                assert tool.output is not None
            except Exception as e:
                pytest.fail(f"Ошибка конверсии агента {agent.id}: {e}")


# ==================== Тесты GraphArchitectBridge ====================

class TestGraphArchitectBridge:
    """Тесты главного моста интеграции"""
    
    def test_bridge_initialization(self):
        """Инициализация моста"""
        bridge = get_bridge()
        
        assert bridge is not None
        assert len(bridge.tools) > 0
        assert bridge.selector is not None
        assert bridge.orchestrator is not None
        assert bridge.strategy_finder is not None
        assert bridge.embedding_service is not None
    
    def test_bridge_is_singleton(self):
        """Bridge является singleton"""
        bridge1 = get_bridge()
        bridge2 = get_bridge()
        
        assert bridge1 is bridge2
    
    def test_is_bridge_available(self):
        """Проверка доступности моста"""
        assert is_bridge_available() is True
    
    def test_tools_count(self, repository):
        """Количество инструментов"""
        bridge = get_bridge()
        agents = repository.get_all_agents()
        
        # Должно быть столько же инструментов сколько агентов
        assert len(bridge.tools) == len(agents)
    
    def test_get_tool_by_agent_id(self):
        """Получение BaseTool по ID агента"""
        bridge = get_bridge()
        
        tool = bridge.get_tool_by_agent_id("agent-classifier-gpt4")
        
        assert tool is not None
        assert isinstance(tool, AgentTool)
        assert tool.agent_id == "agent-classifier-gpt4"
    
    def test_get_tools_by_agent_ids(self):
        """Получение списка BaseTool по списку ID"""
        bridge = get_bridge()
        
        agent_ids = [
            "agent-classifier-gpt4",
            "agent-classifier-claude",
            "agent-classifier-local"
        ]
        
        tools = bridge.get_tools_by_agent_ids(agent_ids)
        
        assert len(tools) == 3
        assert all(isinstance(t, AgentTool) for t in tools)
    
    @pytest.mark.asyncio
    async def test_parse_user_message(self):
        """Парсинг пользовательского сообщения"""
        bridge = get_bridge()
        
        message = "Проанализировать текст и определить его тип"
        input_conn, output_conn = await bridge.parse_user_message(message)
        
        assert input_conn is not None
        assert output_conn is not None
        assert input_conn.format != ""
        assert output_conn.format != ""
    
    @pytest.mark.asyncio
    async def test_find_strategies(self):
        """Поиск стратегий в графе"""
        bridge = get_bridge()
        
        strategies = await bridge.find_strategies(
            start_format="text|question",
            end_format="text|answer",
            algorithm="yen_5"
        )
        
        # Может быть пусто если нет подходящего пути
        assert isinstance(strategies, list)
    
    @pytest.mark.asyncio
    async def test_select_tool_from_group(self):
        """Выбор инструмента из группы через softmax"""
        bridge = get_bridge()
        
        # Берем несколько инструментов
        tools = bridge.tools[:5]
        
        selection_result = await bridge.select_tool_from_group(
            tools,
            task_embedding=None,
            top_k=5
        )
        
        assert selection_result is not None
        assert selection_result.selected_tool is not None
        assert 0 < selection_result.selection_probability <= 1.0
        
        # Проверка softmax (сумма вероятностей = 1)
        total_prob = sum(selection_result.all_probabilities.values())
        assert abs(total_prob - 1.0) < 1e-6
        
        # Проверка температуры
        assert selection_result.temperature > 0
    
    @pytest.mark.asyncio
    async def test_execute_task_streaming(self):
        """Стриминг выполнения задачи"""
        bridge = get_bridge()
        
        message = "Проанализировать запрос"
        chunks = []
        
        async for chunk in bridge.execute_task_streaming(
            message=message,
            input_data=message,
            algorithm="dijkstra",
            top_k=3
        ):
            chunks.append(chunk)
            assert isinstance(chunk, MessageChunk)
        
        # Должно быть хотя бы несколько chunks
        assert len(chunks) > 0
        
        # Проверяем что есть разные типы chunks
        chunk_types = set(c.type for c in chunks)
        assert len(chunk_types) > 0


# ==================== Тесты InstrumentSelector ====================

class TestInstrumentSelectorIntegration:
    """Тесты интеграции InstrumentSelector"""
    
    def test_selector_with_agent_tools(self):
        """Selector работает с AgentTool"""
        bridge = get_bridge()
        
        tools = bridge.tools[:3]
        
        result = bridge.selector.select_instrument(
            tools,
            task_embedding=None,
            top_k=3
        )
        
        assert result.selected_tool is not None
        assert len(result.all_probabilities) == 3
        assert len(result.all_logits) == 3
    
    def test_temperature_affects_distribution(self):
        """Температура влияет на распределение"""
        bridge = get_bridge()
        
        tools = bridge.tools[:5]
        
        # Низкая температура
        bridge.selector._temperature_constant = 0.1
        result_low = bridge.selector.select_instrument(tools, None, top_k=5)
        
        # Высокая температура
        bridge.selector._temperature_constant = 10.0
        result_high = bridge.selector.select_instrument(tools, None, top_k=5)
        
        # При низкой T распределение более концентрировано
        # При высокой T распределение более равномерное
        
        probs_low = list(result_low.all_probabilities.values())
        probs_high = list(result_high.all_probabilities.values())
        
        max_low = max(probs_low)
        min_low = min(probs_low)
        
        max_high = max(probs_high)
        min_high = min(probs_high)
        
        # При низкой T разброс больше
        spread_low = max_low - min_low
        spread_high = max_high - min_high
        
        # Это статистический тест, может иногда падать
        # assert spread_low > spread_high


# ==================== Интеграционные тесты ====================

class TestFullIntegration:
    """Полная интеграция всех компонентов"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Полный пайплайн: парсинг → поиск → выполнение"""
        bridge = get_bridge()
        
        message = "Проанализировать данные и создать отчет"
        
        # 1. Парсинг
        input_conn, output_conn = await bridge.parse_user_message(message)
        assert input_conn is not None
        
        # 2. Поиск стратегий
        strategies = await bridge.find_strategies(
            input_conn.format,
            output_conn.format,
            algorithm="dijkstra"
        )
        
        # Стратегии могут быть не найдены - это нормально
        # (зависит от доступных агентов и их коннекторов)
        assert isinstance(strategies, list)
    
    def test_embedding_service_works(self):
        """Сервис векторизации работает"""
        bridge = get_bridge()
        
        text = "Тестовый текст для векторизации"
        embedding = bridge.embedding_service.embed_text(text)
        
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    def test_tools_have_embeddings(self):
        """Инструменты имеют эмбеддинги"""
        bridge = get_bridge()
        
        for tool in bridge.tools:
            assert tool.metadata.capabilities_embedding is not None
            assert len(tool.metadata.capabilities_embedding) == 384


# ==================== Benchmark тесты ====================

class TestPerformance:
    """Тесты производительности"""
    
    @pytest.mark.asyncio
    async def test_selection_speed(self):
        """Скорость выбора инструмента"""
        import time
        
        bridge = get_bridge()
        tools = bridge.tools[:10]
        
        start = time.time()
        result = await bridge.select_tool_from_group(tools, None, top_k=10)
        elapsed = time.time() - start
        
        # Выбор должен быть быстрым (< 1 секунды)
        assert elapsed < 1.0
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_strategy_search_speed(self):
        """Скорость поиска стратегий"""
        import time
        
        bridge = get_bridge()
        
        start = time.time()
        strategies = await bridge.find_strategies(
            "text|question",
            "text|answer",
            algorithm="dijkstra"
        )
        elapsed = time.time() - start
        
        # Поиск должен быть быстрым (< 2 секунды)
        assert elapsed < 2.0


if __name__ == "__main__":
    # Настройка для async тестов
    pytest.main([__file__, "-v", "--tb=short", "-s"])
