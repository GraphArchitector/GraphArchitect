"""Элемент датасета для обучения"""

from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from uuid import UUID

# Используем TYPE_CHECKING для избежания циклических импортов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...entities.base_tool import BaseTool
    from ..selection.gradient_trace import GradientTrace
    from ..feedback.feedback_data import FeedbackData


@dataclass
class TrainingDatasetItem:
    """
    Элемент датасета для обучения инструментов.
    
    Содержит всю информацию о выполнении задачи:
    - Выбранные инструменты
    - Градиентные трассы
    - Оценки качества
    - Метрики выполнения
    """
    
    # Идентификатор задачи
    task_id: UUID = None
    
    # Описание задачи
    task_description: Optional[str] = None
    
    # Эмбеддинг задачи
    task_embedding: Optional[List[float]] = None
    
    # Область знаний
    domain: str = "general"
    
    # Выбранные инструменты (цепочка выполнения)
    selected_tools: List['BaseTool'] = field(default_factory=list)
    
    # Градиентные трассы для каждого шага
    gradient_traces: List['GradientTrace'] = field(default_factory=list)
    
    # Оценка качества (0-1)
    quality_score: float = 0.5
    
    # Время выполнения (секунды)
    execution_time: float = 0.0
    
    # Стоимость выполнения
    total_cost: float = 0.0
    
    # Обратная связь
    feedbacks: List['FeedbackData'] = field(default_factory=list)
    
    # Время создания
    created_at: datetime = field(default_factory=datetime.utcnow)
