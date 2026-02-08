"""Статистика обучения"""

from dataclasses import dataclass


@dataclass
class TrainingStatistics:
    """Статистика обучения инструментов"""
    
    # Общее количество выполнений
    total_executions: int = 0
    
    # Среднее качество
    average_quality: float = 0.0
    
    # Процент успешных выполнений
    success_rate: float = 0.0
    
    # Среднее время выполнения (секунды)
    average_execution_time: float = 0.0
    
    # Средняя стоимость
    average_cost: float = 0.0
