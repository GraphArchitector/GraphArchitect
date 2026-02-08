"""Базовый класс ребра графа"""

from typing import Generic, TypeVar


T = TypeVar('T')


class BaseEdge:
    """
    Базовое ребро взвешенного графа.
    
    Ребро соединяет две вершины (start_v и end_v) и имеет вес (w).
    """
    
    def __init__(self, start_v: int = 0, end_v: int = 0, w: float = 0.0):
        self.start_v = start_v  # Начальная вершина
        self.end_v = end_v      # Конечная вершина
        self.w = w              # Вес ребра
    
    def either(self) -> int:
        """Получить одну из вершин ребра (обычно начальную)"""
        return self.start_v
    
    def other(self, vertex: int) -> int:
        """
        Получить другую вершину ребра.
        
        Args:
            vertex: Одна из вершин ребра
            
        Returns:
            Другая вершина
        """
        if vertex == self.start_v:
            return self.end_v
        elif vertex == self.end_v:
            return self.start_v
        else:
            raise ValueError(f"Вершина {vertex} не принадлежит ребру")
    
    def __repr__(self):
        return f"Edge({self.start_v} -> {self.end_v}, w={self.w:.2f})"
    
    def __eq__(self, other):
        if not isinstance(other, BaseEdge):
            return False
        return (self.start_v == other.start_v and 
                self.end_v == other.end_v and 
                abs(self.w - other.w) < 1e-9)
    
    def __hash__(self):
        return hash((self.start_v, self.end_v, self.w))
