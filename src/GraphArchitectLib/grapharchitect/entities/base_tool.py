"""Базовый класс инструмента - основная единица в графе"""

from typing import List, Optional
import math
from abc import ABC, abstractmethod

from .tool_metadata import ToolMetadata
from .connectors.connector import Connector
from .quality_functions import QualityFunctions, QualityFunctionParams


class BaseTool(ABC):
    """
    Базовый класс инструмента (ранее BaseAgent).
    
    Инструмент преобразует данные из формата входного коннектора
    в формат выходного коннектора. В графе инструменты находятся
    на ребрах между вершинами-коннекторами.
    
    Содержит функции качества из отчета:
    - f_Q(x) - качество (формула 6.1)
    - f_C(t,e) - стоимость (формула 6.2)
    - f_t(t,e) - время (формула 6.3)
    """
    
    def __init__(self):
        self.input: Connector = Connector()
        self.output: Connector = Connector()
        self.metadata: ToolMetadata = ToolMetadata()
        
        # Функции качества (из отчета, раздел 6)
        self._quality_params = QualityFunctionParams()
        self._quality_functions = QualityFunctions(self._quality_params)
    
    @property
    def reputation(self) -> float:
        """Репутация инструмента (0-1)"""
        return self.metadata.reputation
    
    @reputation.setter
    def reputation(self, value: float):
        self.metadata.reputation = value
    
    @property
    def mean_cost(self) -> float:
        """Средняя стоимость использования"""
        return self.metadata.mean_cost
    
    @mean_cost.setter
    def mean_cost(self, value: float):
        self.metadata.mean_cost = value
    
    def get_graph_weight(self) -> float:
        """
        Расчет веса для графа (LogLoss).
        Чем выше репутация, тем меньше вес (лучше путь).
        
        Returns:
            Вес ребра графа
        """
        safe_reward = max(self.metadata.reputation, 1e-6)
        return -math.log(safe_reward)
    
    def get_logit(self, task_embedding: Optional[List[float]] = None) -> float:
        """
        Вычисление логита (ненормированной оценки качества).
        
        Если вектор theta обучен: f_Q^RL(x) = dot(x, theta) + b (формула 6.1)
        Иначе: cosine_similarity(задача, возможности) + log(репутация)
        
        Args:
            task_embedding: Векторное представление задачи
            
        Returns:
            Логит инструмента для данной задачи
        """
        # Если есть обученные параметры (theta), используем f_Q^RL
        if (task_embedding is not None and 
            self._quality_params.theta is not None and
            len(self._quality_params.theta) > 0):
            return self._quality_functions.f_q_rl(task_embedding)
        
        # Fallback: cosine similarity + reputation
        if task_embedding is None or self.metadata.capabilities_embedding is None:
            return math.log(max(self.metadata.reputation, 1e-6))
        
        similarity = self._cosine_similarity(
            task_embedding, 
            self.metadata.capabilities_embedding
        )
        reputation_bonus = math.log(max(self.metadata.reputation, 1e-6))
        
        return similarity + reputation_bonus
    
    def get_cost(self, input_text: str = "", embedding: Optional[List[float]] = None) -> float:
        """
        Вычисление стоимости (формула 6.2).
        
        f_C(t,e) = C_in * |t| + C_out * f_PT(t,e)
        
        Args:
            input_text: Входной текст
            embedding: Эмбеддинг задачи
            
        Returns:
            Стоимость в у.е.
        """
        # Оценка количества токенов (приблизительно 1 токен = 3 символа)
        tokens_input = len(input_text) // 3 if input_text else 10
        
        return self._quality_functions.f_c(tokens_input, embedding)
    
    def get_duration(self, input_text: str = "", embedding: Optional[List[float]] = None) -> float:
        """
        Вычисление времени (формула 6.3).
        
        f_t(t,e) = k_t * sum((t + n)^alpha)
        
        Args:
            input_text: Входной текст
            embedding: Эмбеддинг задачи
            
        Returns:
            Время в секундах
        """
        tokens_input = len(input_text) // 3 if input_text else 10
        
        return self._quality_functions.f_t(tokens_input, embedding)
    
    def get_full_quality(self, task_embedding: List[float]) -> float:
        """
        Полная оценка качества (формула 6.8).
        
        Комбинирует RL обученную и предобученную оценки.
        
        Args:
            task_embedding: Эмбеддинг задачи
            
        Returns:
            Полная оценка качества
        """
        return self._quality_functions.f_q_full(task_embedding)
    
    def update_quality_params(
        self,
        task_embedding: List[float],
        reward: float,
        learning_rate: float = 0.01
    ):
        """
        Обновить параметры theta (Policy Gradient для f_Q^RL).
        
        Args:
            task_embedding: Эмбеддинг задачи
            reward: Вознаграждение (advantage)
            learning_rate: Скорость обучения
        """
        self._quality_functions.update_theta(task_embedding, reward, learning_rate)
    
    def update_cost_params(
        self,
        tokens_input: int,
        real_output_len: int,
        real_duration: float
    ):
        """
        Обновить параметры стоимости и длительности (SGD).
        
        Args:
            tokens_input: Входные токены
            real_output_len: Реальная длина выхода
            real_duration: Реальная длительность
        """
        self._quality_functions.update_mean_output_len(real_output_len)
        self._quality_functions.update_duration_coefficients(
            tokens_input, real_output_len, real_duration
        )
    
    def get_temperature(self, constant_c: float = 1.0) -> float:
        """
        Вычисление температуры для данного инструмента.
        
        Формула: T_k = C * sqrt(D_k* / m_k)
        где:
        - C - константа
        - D_k* - оценка дисперсии оценок инструмента k
        - m_k - объем выборки для обучения инструмента k
        
        Температура обратно пропорциональна корню от объема выборки.
        
        Args:
            constant_c: Константа C
            
        Returns:
            Температура инструмента
        """
        m_k = max(self.metadata.training_sample_size, 1)
        d_k = max(self.metadata.variance_estimate, 1e-6)
        
        return constant_c * math.sqrt(d_k / m_k)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Косинусное сходство между двумя векторами"""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def clone(self):
        """Клонирование инструмента"""
        import copy
        return copy.deepcopy(self)
    
    @abstractmethod
    def execute(self, input_data) -> any:
        """
        Выполнить инструмент.
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат выполнения
        """
        pass
