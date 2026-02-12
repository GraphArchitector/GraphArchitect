"""Градиентная трассировка для обучения"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...entities.base_tool import BaseTool


@dataclass
class GradientTrace:
    """
    Градиентная информация для обучения.
    
    Сохраняет все необходимые данные для вычисления градиента
    при получении обратной связи (Policy Gradient).
    """
    
    # Эмбеддинг задачи
    task_embedding: Optional[List[float]] = None
    
    # Инструменты-кандидаты
    candidate_tools: List['BaseTool'] = None
    
    # Логиты всех кандидатов
    logits: Dict['BaseTool', float] = None
    
    # Вероятности всех кандидатов (после softmax)
    probabilities: Dict['BaseTool', float] = None
    
    # Выбранный инструмент
    selected_tool: Optional['BaseTool'] = None
    
    # Температура группы
    temperature: float = 1.0
    
    def __post_init__(self):
        if self.candidate_tools is None:
            self.candidate_tools = []
        if self.logits is None:
            self.logits = {}
        if self.probabilities is None:
            self.probabilities = {}
