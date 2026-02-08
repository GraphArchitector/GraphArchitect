"""Сборщик обратной связи"""

from typing import List, Dict
from uuid import UUID
from dataclasses import dataclass, field

from .feedback_data import FeedbackData


@dataclass
class FeedbackCollector:
    """
    Сборщик обратной связи.
    
    Собирает и хранит обратную связь от различных источников
    (пользователи, автоматические критики, системные метрики).
    """
    
    # Хранилище обратной связи: task_id -> список feedback
    _feedbacks: Dict[UUID, List[FeedbackData]] = field(default_factory=dict)
    
    def add_feedback(self, feedback: FeedbackData):
        """
        Добавить обратную связь.
        
        Args:
            feedback: Данные обратной связи
        """
        if feedback.task_id not in self._feedbacks:
            self._feedbacks[feedback.task_id] = []
        
        self._feedbacks[feedback.task_id].append(feedback)
    
    def get_feedbacks(self, task_id: UUID) -> List[FeedbackData]:
        """
        Получить всю обратную связь для задачи.
        
        Args:
            task_id: Идентификатор задачи
            
        Returns:
            Список обратной связи
        """
        return self._feedbacks.get(task_id, [])
    
    def get_all_feedbacks(self) -> Dict[UUID, List[FeedbackData]]:
        """Получить всю обратную связь"""
        return self._feedbacks.copy()
    
    def get_average_quality(self, task_id: UUID) -> float:
        """
        Получить среднюю оценку качества для задачи.
        
        Args:
            task_id: Идентификатор задачи
            
        Returns:
            Средняя оценка качества
        """
        feedbacks = self.get_feedbacks(task_id)
        if not feedbacks:
            return 0.0
        
        return sum(f.quality_score for f in feedbacks) / len(feedbacks)
    
    def clear(self):
        """Очистить все данные"""
        self._feedbacks.clear()
