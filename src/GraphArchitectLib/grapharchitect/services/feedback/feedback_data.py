"""Данные обратной связи"""

from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID


class FeedbackSource(Enum):
    """Источник обратной связи"""
    
    USER = "user"              
    AUTO_CRITIC = "auto_critic" 
    SYSTEM = "system"         


@dataclass
class FeedbackData:
    """
    Данные обратной связи о выполнении задачи.
    
    Используется для дообучения инструментов (пункт 5 из описания системы).
    """
    
    task_id: UUID = None
    
    source: FeedbackSource = FeedbackSource.SYSTEM
    
    success: bool = True
    
    quality_score: float = 0.5
    
    detailed_scores: Dict[str, float] = field(default_factory=dict)

    comment: str = ""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    metadata: Dict[str, any] = field(default_factory=dict)
