"""
Функции качества коннекторов из отчета (раздел 6).

Реализует:
- f_Q(x) - функция качества (6.1)
- f_C(t,e) - функция стоимости (6.2)
- f_t(t,e) - функция времени (6.3)
- Предобучение через PCA + кластеризация (6.4-6.6)
- Итоговая оценка f_q^FULL (6.8)
"""

import math
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QualityFunctionParams:
    """
    Параметры обучаемого коннектора.
    
    Содержит:
    - theta: вектор весов для f_Q^RL
    - b: свободный параметр (bias)
    - api_cost_input: стоимость за входной токен
    - api_cost_output: стоимость за выходной токен
    - k_t: среднее время генерации токена
    - alpha: показатель качества аттеншна (<=1)
    - mean_output_len: средняя длина выхода (обучаемый)
    - duration_coef_input: коэффициент длительности для входа
    - duration_coef_output: коэффициент длительности для выхода
    """
    
    # Параметры f_Q^RL (формула 6.1)
    theta: Optional[List[float]] = None  # Вектор весов
    b: float = 0.0                        # Свободный параметр
    
    # Параметры f_C (формула 6.2)
    api_cost_input: float = 0.001         # Стоимость за входной токен
    api_cost_output: float = 0.002        # Стоимость за выходной токен
    
    # Параметры f_t (формула 6.3)
    k_t: float = 0.01                     # Среднее время генерации токена (секунды)
    alpha: float = 1.0                    # Показатель качества аттеншна (<=1)
    
    # Обучаемые параметры
    mean_output_len: float = 100.0        # Средняя длина выхода (токены)
    max_output_len: float = 4096.0        # Максимальная длина выхода
    adaptive_coef: float = 0.9            # Коэффициент адаптивного сглаживания
    duration_lr: float = 0.01             # Learning rate для коэффициентов длительности
    duration_coef_input: float = 0.001    # Обучаемый коэффициент
    duration_coef_output: float = 0.002   # Обучаемый коэффициент
    
    # Параметры предобучения (формула 6.8)
    p_a: float = 0.5                      # Вес доверия к предобученной оценке
    se_rl: float = 1.0                    # Значимость RL оценки
    se_st: float = 1.0                    # Значимость предобученной оценки
    
    # Предобученная функция (гауссово распределение)
    pretrained_mu: Optional[List[float]] = None    # Среднее
    pretrained_sigma: Optional[List[float]] = None  # Дисперсия


class QualityFunctions:
    """
    Функции качества коннекторов.
    
    Реализует формулы (6.1)-(6.8) из отчета.
    """
    
    def __init__(self, params: QualityFunctionParams):
        """
        Инициализация.
        
        Args:
            params: Параметры функций качества
        """
        self.params = params
    
    def f_q_rl(self, x: List[float]) -> float:
        """
        Функция качества с RL обучением (формула 6.1).
        
        f_Q^RL(x) = dot(x, theta) + b
        
        Args:
            x: Вектор (эмбеддинг задачи)
            
        Returns:
            Оценка качества
        """
        if self.params.theta is None or len(self.params.theta) == 0:
            return self.params.b
        
        # Скалярное произведение x и theta
        if len(x) != len(self.params.theta):
            # Обрезаем или дополняем до нужной длины
            min_len = min(len(x), len(self.params.theta))
            dot_product = sum(
                x[i] * self.params.theta[i] 
                for i in range(min_len)
            )
        else:
            dot_product = sum(
                xi * ti 
                for xi, ti in zip(x, self.params.theta)
            )
        
        return dot_product + self.params.b
    
    def f_c(self, tokens_input: int, embedding: Optional[List[float]] = None) -> float:
        """
        Функция стоимости (формула 6.2).
        
        f_C(t,e) = C_in * |t| + C_out * f_PT(t,e)
        
        Args:
            tokens_input: Количество входных токенов
            embedding: Эмбеддинг (для прогнозирования выходных токенов)
            
        Returns:
            Стоимость в у.е.
        """
        # Прогноз выходных токенов
        predicted_output = self._predict_output_tokens(tokens_input, embedding)
        
        # f_C = C_in * |t| + C_out * f_PT
        cost = (
            self.params.api_cost_input * tokens_input + 
            self.params.api_cost_output * predicted_output
        )
        
        return cost
    
    def f_t(self, tokens_input: int, embedding: Optional[List[float]] = None) -> float:
        """
        Функция времени (формула 6.3).
        
        f_t(t,e) = k_t * sum_{n=0}^{f_PT-1} (t + n)^alpha
        
        Args:
            tokens_input: Количество входных токенов
            embedding: Эмбеддинг
            
        Returns:
            Время в секундах
        """
        predicted_output = self._predict_output_tokens(tokens_input, embedding)
        
        # f_t = k_t * sum((t + n)^alpha) для n от 0 до predicted_output-1
        time_sum = 0.0
        alpha = self.params.alpha
        
        # Оптимизация: для alpha=1 это арифметическая прогрессия
        if abs(alpha - 1.0) < 0.001:
            # sum = predicted_output * tokens_input + predicted_output*(predicted_output-1)/2
            n = int(predicted_output)
            time_sum = n * tokens_input + n * (n - 1) / 2
        else:
            # Общий случай
            for n in range(int(min(predicted_output, 1000))):  # Ограничение для скорости
                time_sum += (tokens_input + n) ** alpha
        
        return self.params.k_t * time_sum
    
    def _predict_output_tokens(
        self, 
        tokens_input: int, 
        embedding: Optional[List[float]] = None
    ) -> float:
        """
        Прогнозирование количества выходных токенов f_PT(t, e).
        
        Простая регрессия. В будущем можно заменить на нейросеть.
        
        Args:
            tokens_input: Входные токены
            embedding: Эмбеддинг
            
        Returns:
            Прогноз выходных токенов
        """
        # Базовый прогноз: средняя длина с ограничением
        predicted = min(self.params.mean_output_len, self.params.max_output_len)
        
        return max(predicted, 1.0)
    
    def f_q_pretrained(self, x: List[float]) -> float:
        """
        Предобученная функция качества (формула 6.7).
        
        f_q^ST(x) = exp(-0.5 * sum((x_m - mu_m)^2 / sigma_m^2))
        
        Ненормированное гауссово распределение.
        
        Args:
            x: Вектор (эмбеддинг задачи)
            
        Returns:
            Предобученная оценка качества
        """
        mu = self.params.pretrained_mu
        sigma = self.params.pretrained_sigma
        
        if mu is None or sigma is None:
            return 0.5  # Нейтральная оценка
        
        min_len = min(len(x), len(mu), len(sigma))
        
        exponent = 0.0
        for m in range(min_len):
            s = max(sigma[m], 1e-6)  # Защита от деления на ноль
            exponent += ((x[m] - mu[m]) / s) ** 2
        
        return math.exp(-0.5 * exponent)
    
    def f_q_full(self, x: List[float]) -> float:
        """
        Полная функция качества (формула 6.8).
        
        f_q^FULL = (1-p_A) * (se_ST / (se_ST + se_RL)) * f_q^RL(x) 
                 + p_A * (1 - se_ST / (se_ST + se_RL)) * f_q^ST(x)
        
        Комбинирует RL обученную и предобученную оценки.
        
        Args:
            x: Вектор (эмбеддинг задачи)
            
        Returns:
            Полная оценка качества
        """
        p_a = self.params.p_a
        se_st = max(self.params.se_st, 1e-6)
        se_rl = max(self.params.se_rl, 1e-6)
        
        # Вес предобученной оценки
        st_weight = se_st / (se_st + se_rl)
        
        # RL оценка
        q_rl = self.f_q_rl(x)
        
        # Предобученная оценка
        q_st = self.f_q_pretrained(x)
        
        # Комбинация (формула 6.8)
        full_quality = (
            (1 - p_a) * st_weight * q_rl + 
            p_a * (1 - st_weight) * q_st
        )
        
        return full_quality
    
    def update_mean_output_len(self, real_output_len: float):
        """
        Обновить среднюю длину выхода (экспоненциальное сглаживание).
        
        mean_len = adaptive_coef * mean_len + (1 - adaptive_coef) * real_len
        
        Args:
            real_output_len: Реальная длина выхода
        """
        self.params.mean_output_len = (
            self.params.adaptive_coef * self.params.mean_output_len + 
            (1 - self.params.adaptive_coef) * real_output_len
        )
    
    def update_duration_coefficients(
        self, 
        tokens_input: int, 
        real_output_len: int, 
        real_duration: float
    ):
        """
        Обновить коэффициенты длительности (SGD).
        
        predicted = coef_in * inp_len + coef_out * out_len
        error = predicted - real_duration
        coef -= lr * error * feature
        
        Args:
            tokens_input: Входные токены
            real_output_len: Реальная длина выхода
            real_duration: Реальная длительность (секунды)
        """
        # Предсказанная длительность
        predicted = (
            self.params.duration_coef_input * tokens_input + 
            self.params.duration_coef_output * real_output_len
        )
        
        # Ошибка
        error = predicted - real_duration
        
        # SGD обновление
        lr = self.params.duration_lr
        self.params.duration_coef_input -= lr * error * tokens_input
        self.params.duration_coef_output -= lr * error * real_output_len
    
    def update_theta(
        self, 
        x: List[float], 
        reward: float, 
        learning_rate: float = 0.01
    ):
        """
        Обновить вектор весов theta (Policy Gradient).
        
        theta += lr * reward * x
        b += lr * reward
        
        Args:
            x: Вектор задачи
            reward: Вознаграждение (advantage)
            learning_rate: Скорость обучения
        """
        if self.params.theta is None:
            self.params.theta = [0.0] * len(x)
        
        # Расширяем theta если нужно
        while len(self.params.theta) < len(x):
            self.params.theta.append(0.0)
        
        # Обновление theta
        for i in range(len(x)):
            if i < len(self.params.theta):
                self.params.theta[i] += learning_rate * reward * x[i]
        
        # Обновление bias
        self.params.b += learning_rate * reward
    
    def compute_significance(self) -> float:
        """
        Вычислить значимость весов (формула 6.6).
        
        s_e ~ sqrt(|w| / N)
        
        Returns:
            Значимость
        """
        if self.params.theta is None:
            return 1.0
        
        # |w| - сумма абсолютных весов
        w_sum = sum(abs(t) for t in self.params.theta)
        N = len(self.params.theta)
        
        if N == 0:
            return 1.0
        
        return math.sqrt(w_sum / N)
