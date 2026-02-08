"""Интерфейс сервиса векторизации"""

from abc import ABC, abstractmethod
from typing import List

# Используем TYPE_CHECKING для избежания циклических импортов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...entities.base_tool import BaseTool


class EmbeddingService(ABC):
    """
    Интерфейс сервиса векторизации.
    
    Преобразует текст в векторные представления (эмбеддинги)
    для вычисления семантической близости.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Создать эмбеддинг для текстового описания задачи.
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Векторное представление текста
        """
        pass
    
    @abstractmethod
    def embed_tool_capabilities(self, tool: 'BaseTool') -> List[float]:
        """
        Создать эмбеддинг для возможностей инструмента.
        
        Args:
            tool: Инструмент
            
        Returns:
            Векторное представление возможностей
        """
        pass
    
    @abstractmethod
    def compute_similarity(
        self,
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """
        Вычислить косинусное сходство между векторами.
        
        Args:
            vector1: Первый вектор
            vector2: Второй вектор
            
        Returns:
            Косинусное сходство (0-1)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Размерность эмбеддингов"""
        pass
