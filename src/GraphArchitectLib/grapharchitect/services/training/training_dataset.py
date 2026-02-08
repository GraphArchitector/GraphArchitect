"""Датасет для обучения"""

from typing import List
from dataclasses import dataclass, field

from .training_dataset_item import TrainingDatasetItem


@dataclass
class TrainingDataset:
    """
    Датасет для обучения инструментов.
    
    Хранит историю выполнений с метриками и обратной связью
    для последующего дообучения.
    """
    
    # Элементы датасета
    _items: List[TrainingDatasetItem] = field(default_factory=list)
    
    def add_item(self, item: TrainingDatasetItem):
        """Добавить элемент в датасет"""
        self._items.append(item)
    
    def get_items(self) -> List[TrainingDatasetItem]:
        """Получить все элементы"""
        return self._items
    
    def get_items_by_quality_threshold(
        self,
        threshold: float
    ) -> List[TrainingDatasetItem]:
        """
        Получить элементы с качеством выше порога.
        
        Args:
            threshold: Минимальная оценка качества
            
        Returns:
            Отфильтрованный список элементов
        """
        return [
            item for item in self._items
            if item.quality_score >= threshold
        ]
    
    def get_items_by_domain(self, domain: str) -> List[TrainingDatasetItem]:
        """
        Получить элементы по области знаний.
        
        Args:
            domain: Область знаний
            
        Returns:
            Отфильтрованный список элементов
        """
        return [
            item for item in self._items
            if item.domain == domain
        ]
    
    def clear(self):
        """Очистить датасет"""
        self._items.clear()
    
    def size(self) -> int:
        """Получить размер датасета"""
        return len(self._items)
