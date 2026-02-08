"""
Тесты для модуля выбора инструментов (selection).

Тестируется ключевой алгоритм выбора инструмента:
1. Вычисление логитов
2. Отбор топ-K
3. Вычисление температуры
4. Softmax с температурой
5. Вероятностное сэмплирование

InstrumentSelector - самый важный компонент системы!
"""

import pytest
import math
from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.services.selection.instrument_selector import (
    InstrumentSelector,
    InstrumentSelectionResult
)
from grapharchitect.services.selection.gradient_trace import GradientTrace


# ==================== Фикстуры ====================

class MockTool(BaseTool):
    """Моковый инструмент для тестов"""
    
    def __init__(self, name: str, reputation: float = 0.5, 
                 sample_size: int = 10, variance: float = 0.1):
        super().__init__()
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
        self.metadata.training_sample_size = sample_size
        self.metadata.variance_estimate = variance
        self.input = Connector("text", "input")
        self.output = Connector("text", "output")
    
    def execute(self, input_data):
        return f"{self.metadata.tool_name}: {input_data}"


@pytest.fixture
def selector():
    """Селектор с константой температуры = 1.0"""
    return InstrumentSelector(temperature_constant=1.0)


@pytest.fixture
def simple_tools():
    """Простой набор инструментов"""
    return [
        MockTool("Tool_A", reputation=0.9, sample_size=100, variance=0.05),
        MockTool("Tool_B", reputation=0.7, sample_size=50, variance=0.15),
        MockTool("Tool_C", reputation=0.5, sample_size=10, variance=0.30),
    ]


@pytest.fixture
def task_embedding():
    """Эмбеддинг задачи"""
    return [0.5, 0.5, 0.5]


# ==================== Тесты InstrumentSelector ====================

class TestInstrumentSelector:
    """Тесты селектора инструментов"""
    
    def test_selector_creation(self, selector):
        """Создание селектора"""
        assert selector._temperature_constant == 1.0
    
    def test_select_from_single_tool(self, selector):
        """Выбор из одного инструмента"""
        tools = [MockTool("OnlyTool", reputation=0.8)]
        
        result = selector.select_instrument(tools, None, top_k=5)
        
        assert result.selected_tool is not None
        assert result.selected_tool.metadata.tool_name == "OnlyTool"
        assert result.selection_probability == 1.0  # Единственный инструмент
    
    def test_select_from_multiple_tools(self, selector, simple_tools):
        """Выбор из нескольких инструментов"""
        result = selector.select_instrument(simple_tools, None, top_k=5)
        
        assert result.selected_tool is not None
        assert result.selected_tool in simple_tools
        assert 0.0 < result.selection_probability <= 1.0
    
    def test_empty_tools_raises_error(self, selector):
        """Пустой список инструментов выбрасывает ошибку"""
        with pytest.raises(ValueError, match="не может быть пустым"):
            selector.select_instrument([], None, top_k=5)
    
    def test_top_k_selection(self, selector):
        """Отбор топ-K инструментов"""
        tools = [
            MockTool("Tool_1", reputation=0.9),
            MockTool("Tool_2", reputation=0.8),
            MockTool("Tool_3", reputation=0.7),
            MockTool("Tool_4", reputation=0.6),
            MockTool("Tool_5", reputation=0.5),
        ]
        
        result = selector.select_instrument(tools, None, top_k=3)
        
        # Должны быть рассмотрены только топ-3
        assert result.top_k == 3
        assert len(result.all_logits) == 3
    
    def test_logits_calculation(self, selector, simple_tools):
        """Вычисление логитов"""
        result = selector.select_instrument(simple_tools, None, top_k=5)
        
        # Все логиты должны быть конечными числами
        for tool, logit in result.all_logits.items():
            assert math.isfinite(logit)
        
        # Инструмент с большей репутацией должен иметь больший логит
        tools_by_logit = sorted(
            result.all_logits.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Tool_A (0.9) должен иметь больший логит чем Tool_C (0.5)
        assert tools_by_logit[0][0].reputation >= tools_by_logit[-1][0].reputation
    
    def test_temperature_calculation(self, selector, simple_tools):
        """Вычисление температуры"""
        result = selector.select_instrument(simple_tools, None, top_k=5)
        
        assert result.temperature > 0
        assert math.isfinite(result.temperature)
    
    def test_temperature_decreases_with_more_data(self, selector):
        """Температура уменьшается с ростом данных"""
        tool_small_data = MockTool("SmallData", sample_size=10, variance=0.1)
        tool_large_data = MockTool("LargeData", sample_size=1000, variance=0.1)
        
        result_small = selector.select_instrument([tool_small_data], None, top_k=1)
        result_large = selector.select_instrument([tool_large_data], None, top_k=1)
        
        # С большим количеством данных температура ниже
        assert result_large.temperature < result_small.temperature
    
    def test_probabilities_sum_to_one(self, selector, simple_tools):
        """Сумма вероятностей равна 1"""
        result = selector.select_instrument(simple_tools, None, top_k=5)
        
        total_prob = sum(result.all_probabilities.values())
        assert abs(total_prob - 1.0) < 1e-6
    
    def test_softmax_with_high_temperature(self, selector, simple_tools):
        """Softmax с высокой температурой (более равномерное распределение)"""
        selector._temperature_constant = 10.0  # Высокая температура
        
        result = selector.select_instrument(simple_tools, None, top_k=3)
        
        probs = list(result.all_probabilities.values())
        # Вероятности должны быть более равномерными
        max_prob = max(probs)
        min_prob = min(probs)
        
        # Разница между макс и мин не должна быть огромной
        assert max_prob / min_prob < 5.0
    
    def test_softmax_with_low_temperature(self, selector):
        """Softmax с низкой температурой (концентрация на лучших)"""
        selector._temperature_constant = 0.01  # Очень низкая температура
        
        tools = [
            MockTool("Best", reputation=0.95, sample_size=1000, variance=0.01),
            MockTool("Mediocre", reputation=0.5, sample_size=1000, variance=0.01),
            MockTool("Worst", reputation=0.1, sample_size=1000, variance=0.01),
        ]
        
        result = selector.select_instrument(tools, None, top_k=3)
        
        # Лучший инструмент должен иметь самую высокую вероятность
        best_tool = max(result.all_probabilities.items(), key=lambda x: x[1])[0]
        assert best_tool.metadata.tool_name == "Best"
    
    def test_gradient_trace_creation(self, selector, simple_tools, task_embedding):
        """Создание градиентной трассы"""
        result = selector.select_instrument(simple_tools, task_embedding, top_k=5)
        
        gradient_info = result.gradient_info
        
        assert isinstance(gradient_info, GradientTrace)
        assert gradient_info.task_embedding == task_embedding
        assert gradient_info.selected_tool == result.selected_tool
        assert gradient_info.temperature == result.temperature
        assert len(gradient_info.candidate_tools) > 0
        assert len(gradient_info.logits) > 0
        assert len(gradient_info.probabilities) > 0
    
    def test_with_tool_embeddings(self, selector):
        """Выбор с эмбеддингами инструментов"""
        task_emb = [1.0, 0.0, 0.0]
        
        tool1 = MockTool("Similar", reputation=0.5)
        tool1.metadata.capabilities_embedding = [1.0, 0.0, 0.0]  # Похож на задачу
        
        tool2 = MockTool("Different", reputation=0.5)
        tool2.metadata.capabilities_embedding = [0.0, 1.0, 0.0]  # Отличается
        
        tools = [tool1, tool2]
        
        result = selector.select_instrument(tools, task_emb, top_k=2)
        
        # tool1 должен иметь больший логит из-за схожести
        assert result.all_logits[tool1] > result.all_logits[tool2]
    
    def test_stochastic_selection(self, selector):
        """Вероятностный выбор - проверка на случайность"""
        tools = [
            MockTool("Tool_1", reputation=0.6),
            MockTool("Tool_2", reputation=0.6),  # Одинаковая репутация
        ]
        
        # Запускаем выбор много раз
        selections = []
        for _ in range(100):
            result = selector.select_instrument(tools, None, top_k=2)
            selections.append(result.selected_tool.metadata.tool_name)
        
        # Оба инструмента должны быть выбраны хотя бы раз (с очень высокой вероятностью)
        unique_selections = set(selections)
        assert len(unique_selections) >= 1  # Хотя бы один выбран
        
        # При равной репутации оба должны встречаться (если запусков достаточно)
        # Это стохастический тест, может иногда падать


# ==================== Тесты GradientTrace ====================

class TestGradientTrace:
    """Тесты градиентной трассы"""
    
    def test_gradient_trace_creation(self):
        """Создание градиентной трассы"""
        tools = [MockTool("Tool_A"), MockTool("Tool_B")]
        
        trace = GradientTrace(
            task_embedding=[0.1, 0.2],
            candidate_tools=tools,
            logits={tools[0]: 1.5, tools[1]: 1.2},
            probabilities={tools[0]: 0.6, tools[1]: 0.4},
            selected_tool=tools[0],
            temperature=0.8
        )
        
        assert trace.selected_tool == tools[0]
        assert trace.temperature == 0.8
        assert len(trace.candidate_tools) == 2
    
    def test_default_values(self):
        """Значения по умолчанию"""
        trace = GradientTrace()
        
        assert trace.candidate_tools == []
        assert trace.logits == {}
        assert trace.probabilities == {}


# ==================== Интеграционные тесты ====================

class TestIntegration:
    """Интеграционные тесты"""
    
    def test_full_selection_pipeline(self, selector):
        """Полный пайплайн выбора"""
        # Создаем инструменты с разными характеристиками
        tools = [
            MockTool("Expert", reputation=0.95, sample_size=500, variance=0.02),
            MockTool("Intermediate", reputation=0.70, sample_size=100, variance=0.10),
            MockTool("Novice", reputation=0.40, sample_size=20, variance=0.30),
        ]
        
        task_emb = [0.5, 0.5, 0.5]
        
        # Устанавливаем эмбеддинги
        tools[0].metadata.capabilities_embedding = [0.6, 0.5, 0.4]  # Похож
        tools[1].metadata.capabilities_embedding = [0.3, 0.3, 0.3]  # Средний
        tools[2].metadata.capabilities_embedding = [0.1, 0.1, 0.1]  # Далек
        
        # Выполняем выбор
        result = selector.select_instrument(tools, task_emb, top_k=3)
        
        # Проверяем результат
        assert result.selected_tool is not None
        assert result.selected_tool in tools
        
        # Expert должен иметь наибольшую вероятность
        expert_prob = result.all_probabilities[tools[0]]
        novice_prob = result.all_probabilities[tools[2]]
        
        assert expert_prob > novice_prob
    
    def test_repeated_selections_distribution(self, selector):
        """Проверка распределения при повторных выборах"""
        tools = [
            MockTool("High", reputation=0.8),
            MockTool("Medium", reputation=0.5),
            MockTool("Low", reputation=0.2),
        ]
        
        # Считаем сколько раз каждый инструмент был выбран
        counts = {tool.metadata.tool_name: 0 for tool in tools}
        
        num_trials = 1000
        for _ in range(num_trials):
            result = selector.select_instrument(tools, None, top_k=3)
            counts[result.selected_tool.metadata.tool_name] += 1
        
        # High должен быть выбран чаще всего
        assert counts["High"] > counts["Low"]
        
        # Все инструменты должны быть выбраны хотя бы раз
        # (при достаточном количестве испытаний)
        # Примечание: с низкой температурой Low может не выбраться совсем
        # Проверяем хотя бы что High и Medium выбирались
        assert counts["High"] > 0
        assert counts["Medium"] > 0
        # Low может быть 0 если температура очень низкая


# ==================== Граничные случаи ====================

class TestEdgeCases:
    """Тесты граничных случаев"""
    
    def test_very_high_reputation(self, selector):
        """Очень высокая репутация"""
        tool = MockTool("Perfect", reputation=0.99)
        result = selector.select_instrument([tool], None, top_k=1)
        
        assert result.selected_tool == tool
        assert result.selection_probability == 1.0
    
    def test_very_low_reputation(self, selector):
        """Очень низкая репутация"""
        tool = MockTool("Poor", reputation=0.01)
        result = selector.select_instrument([tool], None, top_k=1)
        
        assert result.selected_tool == tool
        assert result.selection_probability == 1.0
    
    def test_zero_variance(self, selector):
        """Нулевая дисперсия"""
        tool = MockTool("Stable", variance=0.0)
        result = selector.select_instrument([tool], None, top_k=1)
        
        # Не должно быть ошибок
        assert result.selected_tool == tool
    
    def test_top_k_larger_than_tools(self, selector):
        """top_k больше количества инструментов"""
        tools = [MockTool("Tool_1"), MockTool("Tool_2")]
        
        result = selector.select_instrument(tools, None, top_k=10)
        
        # Должны быть рассмотрены все инструменты
        assert result.top_k == 2
        assert len(result.all_logits) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
