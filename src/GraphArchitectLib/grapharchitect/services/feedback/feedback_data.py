"""Данные обратной связи"""

from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID


class FeedbackSource(Enum):
    """Источник обратной связи"""
    
    USER = "user"              # От пользователя
    AUTO_CRITIC = "auto_critic"  # Автоматический критик
    SYSTEM = "system"          # Системная метрика


@dataclass
class FeedbackData:
    """
    Данные обратной связи о выполнении задачи.
    
    Используется для дообучения инструментов (пункт 5 из описания системы).
    """
    
    # Идентификатор задачи
    task_id: UUID = None
    
    # Источник обратной связи
    source: FeedbackSource = FeedbackSource.SYSTEM
    
    # Успешность выполнения
    success: bool = True
    
    # Оценка качества (0-1)
    quality_score: float = 0.5
    
    # Детализированные оценки по критериям
    detailed_scores: Dict[str, float] = field(default_factory=dict)
    
    # Комментарий
    comment: str = ""
    
    # Время создания
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Дополнительные метаданные
    metadata: Dict[str, any] = field(default_factory=dict)
