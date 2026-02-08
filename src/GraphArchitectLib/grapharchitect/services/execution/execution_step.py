"""Шаг выполнения задачи"""

from typing import List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

# Используем TYPE_CHECKING для избежания циклических импортов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...entities.base_tool import BaseTool
    from ..selection.instrument_selector import InstrumentSelectionResult


@dataclass
class ExecutionStep:
    """
    Один шаг выполнения задачи.
    
    Каждый шаг соответствует обработке одной группы инструментов
    (ребру графа).
    """
    
    # Номер шага
    step_number: int = 0
    
    # Доступные инструменты на этом шаге
    available_tools: List['BaseTool'] = field(default_factory=list)
    
    # Выбранный инструмент
    selected_tool: Optional['BaseTool'] = None
    
    # Результат выбора инструмента
    selection_result: Optional['InstrumentSelectionResult'] = None
    
    # Входные данные
    input_data: Any = None
    
    # Выходные данные
    output_data: Any = None
    
    # Успешность выполнения
    success: bool = False
    
    # Сообщение об ошибке
    error_message: str = ""
    
    # Время выполнения
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    execution_time: float = 0.0  # секунды
    
    # Стоимость выполнения
    cost: float = 0.0
