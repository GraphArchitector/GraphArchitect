"""
Продвинутый селектор инструментов с полной формулой R(x).

R(x) = (t_sc · (w_q · f_q(x) - w_c · f_c(x))) / log₁₀(w_t · f_t(x) + 1)

Где:
- t_sc - масштабный коэффициент
- w_q, w_c, w_t - веса качества, стоимости, времени
- f_q(x) - функция качества (логит)
- f_c(x) - функция стоимости
- f_t(x) - функция времени
"""

import math
import random
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from ...entities.base_tool import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class AdvancedSelectionResult:
    """Результат продвинутого выбора инструмента."""
    
    selected_tool: BaseTool
    selection_probability: float
    temperature: float
    logits: Dict[str, float]
    costs: Dict[str, float]
    times: Dict[str, float]
    r_metrics: Dict[str, float]  # Значения R(x) для каждого
    probabilities: Dict[str, float]


class AdvancedInstrumentSelector:
    """
    Продвинутый селектор с полной формулой R(x).
    
    Учитывает:
    - Качество (через логиты)
    - Стоимость (API cost)
    - Время выполнения
    - Адаптивная температура
    """
    
    def __init__(
        self,
        scale_coefficient: float = 1.0,
        weight_quality: float = 0.5,
        weight_cost: float = 0.25,
        weight_time: float = 0.25,
        temperature_constant: float = 1.0
    ):
        """
        Инициализация селектора.
        
        Args:
            scale_coefficient: Масштабный коэффициент t_sc
            weight_quality: Вес качества w_q
            weight_cost: Вес стоимости w_c
            weight_time: Вес времени w_t
            temperature_constant: Константа температуры C
        """
        self._scale_coef = scale_coefficient
        self._w_quality = weight_quality
        self._w_cost = weight_cost
        self._w_time = weight_time
        self._temperature_constant = temperature_constant
        
        logger.info(
            f"AdvancedInstrumentSelector initialized: "
            f"t_sc={scale_coefficient}, w_q={weight_quality}, "
            f"w_c={weight_cost}, w_t={weight_time}"
        )
    
    def select_instrument(
        self,
        instruments: List[BaseTool],
        task_embedding: List[float],
        top_k: int = 5,
        normalize_costs: bool = True,
        normalize_times: bool = True
    ) -> AdvancedSelectionResult:
        """
        Выбрать инструмент с учетом качества, стоимости и времени.
        
        Args:
            instruments: Список инструментов для выбора
            task_embedding: Эмбеддинг задачи
            top_k: Количество лучших для рассмотрения
            normalize_costs: Нормализовать стоимости в [0,1]
            normalize_times: Нормализовать времена в [0,1]
            
        Returns:
            Результат выбора с метриками
        """
        if not instruments:
            raise ValueError("No instruments provided")
        
        if top_k <= 0:
            top_k = len(instruments)
        
        # Шаг 1: Вычисление логитов (качество)
        logits = {}
        for tool in instruments:
            logit = tool.get_logit(task_embedding)
            tool_id = id(tool)
            logits[tool_id] = logit
        
        # Вычисление стоимостей
        costs = {}
        for tool in instruments:
            cost = tool.metadata.mean_cost
            costs[id(tool)] = cost
        
        # Нормализация стоимостей
        if normalize_costs and costs:
            max_cost = max(costs.values()) if costs.values() else 1.0
            if max_cost > 0:
                costs = {k: v / max_cost for k, v in costs.items()}
        
        # Вычисление времен
        times = {}
        for tool in instruments:
            time = tool.metadata.mean_time_answer
            times[id(tool)] = time
        
        # Нормализация времени выполнения
        if normalize_times and times:
            max_time = max(times.values()) if times.values() else 1.0
            if max_time > 0:
                times = {k: v / max_time for k, v in times.items()}
        
        # Вычисление R(x) для каждого инструмента
        r_metrics = {}
        for tool in instruments:
            tool_id = id(tool)
            
            quality = logits[tool_id]
            cost = costs[tool_id]
            time = times[tool_id]

            # R(x) = (t_sc · (w_q · f_q - w_c · f_c)) / log₁₀(w_t · f_t + 1)
            
            numerator = self._scale_coef * (
                self._w_quality * quality - self._w_cost * cost
            )
            
            denominator = math.log10(self._w_time * time + 1.0)
            if denominator == 0:
                denominator = 1.0
            
            r_value = numerator / denominator
            r_metrics[tool_id] = r_value
        
        # Отбор топ-K по R(x)
        sorted_items = sorted(
            r_metrics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_k_ids = [tool_id for tool_id, _ in sorted_items[:top_k]]
        top_k_tools = [tool for tool in instruments if id(tool) in top_k_ids]
        
        # Вычисление температуры группы
        temperature = self._calculate_group_temperature(top_k_tools)
        
        # Softmax с температурой по R(x)
        probabilities = self._apply_softmax_with_temperature(
            {tool_id: r_metrics[tool_id] for tool_id in top_k_ids},
            temperature
        )
        
        # Вероятностное сэмплирование
        selected_tool = self._sample_from_probabilities(
            top_k_tools,
            probabilities
        )
        
        selected_id = id(selected_tool)
        
        return AdvancedSelectionResult(
            selected_tool=selected_tool,
            selection_probability=probabilities[selected_id],
            temperature=temperature,
            logits=logits,
            costs=costs,
            times=times,
            r_metrics=r_metrics,
            probabilities=probabilities
        )
    
    def _calculate_group_temperature(self, tools: List[BaseTool]) -> float:
        """
        Вычислить адаптивную температуру группы.
        
        """
        if not tools:
            return 1.0
        
        K = len(tools)
        sum_temp = 0.0
        
        for tool in tools:
            variance = tool.metadata.variance_estimate
            sample_size = tool.metadata.training_sample_size
            
            if sample_size > 0:
                temp_k = math.sqrt(variance / sample_size)
            else:
                temp_k = 1.0  # Высокая неопределенность для новых
            
            sum_temp += temp_k
        
        group_temperature = (self._temperature_constant / K) * sum_temp
        
        return max(group_temperature, 0.01)  # Минимум 0.01
    
    def _apply_softmax_with_temperature(
        self,
        r_values: Dict[int, float],
        temperature: float
    ) -> Dict[int, float]:
        """
        Применить softmax с температурой к R(x) значениям.
        
        P(k) = exp(R_k / T) / Σ exp(R_i / T)
        """
        if not r_values:
            return {}
        
        # Для численной стабильности вычитаем максимум
        max_r = max(r_values.values())
        
        exp_values = {}
        for tool_id, r_val in r_values.items():
            exp_val = math.exp((r_val - max_r) / temperature)
            exp_values[tool_id] = exp_val
        
        # Нормализация
        total = sum(exp_values.values())
        
        probabilities = {}
        for tool_id, exp_val in exp_values.items():
            probabilities[tool_id] = exp_val / total
        
        return probabilities
    
    def _sample_from_probabilities(
        self,
        tools: List[BaseTool],
        probabilities: Dict[int, float]
    ) -> BaseTool:
        """
        Вероятностное сэмплирование инструмента.
        
        Args:
            tools: Список инструментов
            probabilities: Вероятности для каждого
            
        Returns:
            Выбранный инструмент
        """
        # Рулетка
        rand_val = random.random()
        cumulative = 0.0
        
        for tool in tools:
            tool_id = id(tool)
            prob = probabilities.get(tool_id, 0.0)
            cumulative += prob
            
            if rand_val <= cumulative:
                return tool
        
        # Fallback на последний (на случай ошибок округления)
        return tools[-1]

    def _sample_tool(self, tools: List[BaseTool], probabilities: Dict[int, float]) -> BaseTool:
        tools_id = list(probabilities.keys())
        weights = list(probabilities.values())
        tool_id = self._random.choices(tools_id, weights=weights, k=1)[0]
        return tools[tool_id]