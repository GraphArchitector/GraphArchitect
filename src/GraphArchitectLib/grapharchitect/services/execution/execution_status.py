"""Статусы выполнения задачи"""

from enum import Enum


class ExecutionStatus(Enum):
    """Статус выполнения задачи"""
    
    PENDING = "pending"      # Ожидает выполнения
    RUNNING = "running"      # Выполняется
    COMPLETED = "completed"  # Завершено успешно
    FAILED = "failed"        # Завершено с ошибкой
    CANCELLED = "cancelled"  # Отменено
