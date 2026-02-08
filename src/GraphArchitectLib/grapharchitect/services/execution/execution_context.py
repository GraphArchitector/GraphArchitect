"""Контекст выполнения задачи"""

from typing import List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from .execution_status import ExecutionStatus
from .execution_step import ExecutionStep
from ..selection.gradient_trace import GradientTrace

# Используем TYPE_CHECKING для избежания циклических импортов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...entities.task_definition import TaskDefinition


@dataclass
class ExecutionContext:
    """
    Контекст выполнения задачи.
    
    Содержит всю информацию о ходе выполнения задачи:
    - Текущее состояние
    - Историю шагов
    - Градиентные трассы для обучения
    - Метрики выполнения
    """
    
    # Идентификатор выполнения
    task_id: UUID = field(default_factory=uuid4)
    
    # Определение задачи
    task: Optional['TaskDefinition'] = None
    
    # Статус выполнения
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    # Входные данные
    input_data: Any = None
    
    # Текущие данные (обновляются на каждом шаге)
    current_data: Any = None
    
    # Финальный результат
    result: Any = None
    
    # Шаги выполнения
    execution_steps: List[ExecutionStep] = field(default_factory=list)
    
    # Градиентные трассы для обучения
    gradient_traces: List[GradientTrace] = field(default_factory=list)
    
    # Метрики
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_time: float = 0.0  # секунды
    total_cost: float = 0.0
    
    # Ошибки
    error_message: str = ""
    
    def add_step(self, step: ExecutionStep):
        """Добавить шаг выполнения"""
        self.execution_steps.append(step)
    
    def add_gradient_trace(self, trace: GradientTrace):
        """Добавить градиентную трассу"""
        self.gradient_traces.append(trace)
    
    def get_total_steps(self) -> int:
        """Получить общее количество шагов"""
        return len(self.execution_steps)
    
    def get_successful_steps(self) -> int:
        """Получить количество успешных шагов"""
        return sum(1 for step in self.execution_steps if step.success)
    
    def is_successful(self) -> bool:
        """Проверить, успешно ли выполнена задача"""
        return self.status == ExecutionStatus.COMPLETED
