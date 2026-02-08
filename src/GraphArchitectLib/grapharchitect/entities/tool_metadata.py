"""Метаданные инструмента - информация о репутации, стоимости и статистике"""

from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ToolMetadata:
    """Метаданные инструмента для обучения и выбора"""
    
    # Основная информация
    tool_name: str = ""
    description: str = ""
    
    # Репутация и качество (от 0 до 1)
    reputation: float = 0.5
    
    # Стоимость выполнения (условные единицы)
    mean_cost: float = 1.0
    
    # Среднее время ответа (секунды)
    mean_time_answer: float = 1.0
    
    # Статистика для дообучения
    training_sample_size: int = 1  # Количество обучающих примеров
    variance_estimate: float = 1.0  # Оценка дисперсии качества
    quality_scores: List[float] = field(default_factory=list)  # История оценок качества
    
    # История выполнения
    execution_history: List['ExecutionRecord'] = field(default_factory=list)
    last_training_date: datetime = field(default_factory=lambda: datetime.min)
    
    # Эмбеддинг возможностей инструмента (векторное представление)
    capabilities_embedding: Optional[List[float]] = None


# Импорт для типизации (избегаем циклических импортов)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .execution_record import ExecutionRecord
