"""Селектор инструментов с логитами, температурой и вероятностным сэмплированием"""

from typing import List, Dict, Optional
import random
import math
from dataclasses import dataclass

from ...entities.base_tool import BaseTool
from .gradient_trace import GradientTrace


@dataclass
class InstrumentSelectionResult:
    """Результат выбора инструмента"""
    
    # Выбранный инструмент
    selected_tool: BaseTool
    
    # Вероятность выбора
    selection_probability: float
    
    # Логиты всех топ-K кандидатов
    all_logits: Dict[BaseTool, float]
    
    # Вероятности всех топ-K кандидатов
    all_probabilities: Dict[BaseTool, float]
    
    # Температура группы
    temperature: float
    
    # Количество рассмотренных кандидатов
    top_k: int
    
    # Градиентная информация для обучения
    gradient_info: GradientTrace


class InstrumentSelector:
    """
    Селектор инструментов с логитами, температурой и вероятностным сэмплированием.
    
    Реализует выбор инструмента из группы:
    1. Вычисление логитов от всех инструментов
    2. Отбор топ-K по логитам
    3. Вычисление температуры группы: T = (C/K) * Σ√(D_k* / m_k)
    4. Применение softmax с температурой
    5. Сэмплирование согласно вероятностям
    """
    
    def __init__(self, temperature_constant: float = 1.0):
        """
        Инициализация селектора.
        
        Args:
            temperature_constant: Константа C для вычисления температуры
        """
        self._temperature_constant = temperature_constant
        self._random = random.Random()
    
    def select_instrument(
        self,
        tools: List[BaseTool],
        task_embedding: Optional[List[float]] = None,
        top_k: int = 5
    ) -> InstrumentSelectionResult:
        """
        Выбрать инструмент из группы с учетом логитов и температуры.
        
        Args:
            tools: Список инструментов-кандидатов
            task_embedding: Векторное представление задачи
            top_k: Количество лучших кандидатов для рассмотрения
            
        Returns:
            Результат выбора инструмента
        """
        if not tools:
            raise ValueError("Список инструментов не может быть пустым")
        
        # 1. Получить логиты от всех инструментов
        logits = self._get_logits(tools, task_embedding)
        
        # 2. Отобрать топ-K по логитам
        top_k_items = sorted(
            logits.items(),
            key=lambda x: x[1],
            reverse=True
        )[:min(top_k, len(tools))]
        
        top_k_tools = [tool for tool, _ in top_k_items]
        top_k_logits = {tool: logit for tool, logit in top_k_items}
        
        # 3. Вычислить температуру группы
        temperature = self._calculate_group_temperature(top_k_tools)
        
        # 4. Применить softmax с температурой
        probabilities = self._apply_softmax_with_temperature(
            top_k_logits,
            temperature
        )
        
        # 5. Сэмплировать инструмент
        selected_tool = self._sample_tool(probabilities)
        
        # 6. Сформировать результат с градиентной информацией
        return InstrumentSelectionResult(
            selected_tool=selected_tool,
            selection_probability=probabilities[selected_tool],
            all_logits=top_k_logits,
            all_probabilities=probabilities,
            temperature=temperature,
            top_k=len(top_k_tools),
            gradient_info=GradientTrace(
                task_embedding=task_embedding,
                candidate_tools=top_k_tools,
                logits=top_k_logits,
                probabilities=probabilities,
                selected_tool=selected_tool,
                temperature=temperature
            )
        )
    
    def _get_logits(
        self,
        tools: List[BaseTool],
        task_embedding: Optional[List[float]]
    ) -> Dict[BaseTool, float]:
        """
        Получить логиты от всех инструментов.
        
        Args:
            tools: Список инструментов
            task_embedding: Эмбеддинг задачи
            
        Returns:
            Словарь {инструмент: логит}
        """
        logits = {}
        for tool in tools:
            logits[tool] = tool.get_logit(task_embedding)
        return logits
    
    def _calculate_group_temperature(self, tools: List[BaseTool]) -> float:
        """
        Вычислить температуру группы.
        
        Формула: T = (C/K) * Σ√(D_k* / m_k)
        где:
        - C - константа
        - K - количество инструментов
        - D_k* - оценка дисперсии оценок инструмента k
        - m_k - объем выборки для обучения инструмента k
        
        Args:
            tools: Список инструментов
            
        Returns:
            Температура группы
        """
        if not tools:
            return 1.0
        
        temperature_sum = sum(
            tool.get_temperature(self._temperature_constant)
            for tool in tools
        )
        
        return temperature_sum / len(tools)
    
    def _apply_softmax_with_temperature(
        self,
        logits: Dict[BaseTool, float],
        temperature: float
    ) -> Dict[BaseTool, float]:
        """
        Применить softmax с температурой.
        
        P(k) = exp(logit_k / T) / Σ exp(logit_i / T)
        
        Args:
            logits: Словарь логитов
            temperature: Температура
            
        Returns:
            Словарь вероятностей
        """
        # Для численной стабильности вычитаем максимальный логит
        max_logit = max(logits.values())
        
        # Вычисляем exp((logit - max_logit) / T)
        exp_values = {
            tool: math.exp((logit - max_logit) / temperature)
            for tool, logit in logits.items()
        }
        
        # Нормализуем
        sum_exp = sum(exp_values.values())
        
        probabilities = {
            tool: exp_val / sum_exp
            for tool, exp_val in exp_values.items()
        }
        
        return probabilities
    
    def _sample_tool(self, probabilities: Dict[BaseTool, float]) -> BaseTool:
        """
        Сэмплировать инструмент согласно распределению вероятностей.
        
        Args:
            probabilities: Словарь вероятностей
            
        Returns:
            Выбранный инструмент
        """
        sample = self._random.random()
        cumulative = 0.0
        
        for tool, prob in probabilities.items():
            cumulative += prob
            if sample <= cumulative:
                return tool
        
        # На случай ошибок округления - возвращаем первый
        return next(iter(probabilities.keys()))
