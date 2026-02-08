"""Запись о выполнении инструмента"""

from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass
from uuid import UUID


@dataclass
class ExecutionRecord:
    """Запись о выполнении инструмента для истории"""
    
    task_id: UUID
    execution_time: datetime
    time_taken: float  # Время выполнения в секундах
    cost: float  # Стоимость выполнения
    quality_score: float  # Оценка качества (0-1)
    task_embedding: Optional[List[float]] = None  # Эмбеддинг задачи
    success: bool = True  # Успешность выполнения
