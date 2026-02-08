"""Дерево кратчайших путей"""

from typing import List, Optional, TypeVar, Generic

from .base_edge import BaseEdge

TEdge = TypeVar('TEdge', bound=BaseEdge)


class ShortestPathTree(Generic[TEdge]):
    """
    Дерево кратчайших путей.
    
    Используется для восстановления путей после выполнения
    алгоритмов Дейкстры или других алгоритмов поиска.
    """
    
    def __init__(self, edges: List[Optional[TEdge]], distances: List[float]):
        """
        Инициализация дерева кратчайших путей.
        
        Args:
            edges: Массив ребер (edge_to[v] - ребро, ведущее в v)
            distances: Массив расстояний до вершин
        """
        self._edges = edges
        self._distances = distances
    
    def has_path_to(self, v: int) -> bool:
        """
        Проверка существования пути до вершины.
        
        Args:
            v: Индекс вершины
            
        Returns:
            True если путь существует
        """
        return self._distances[v] < float('inf')
    
    def distance_to(self, v: int) -> float:
        """
        Получить расстояние до вершины.
        
        Args:
            v: Индекс вершины
            
        Returns:
            Расстояние до вершины
        """
        return self._distances[v]
    
    def get_path(self, v: int) -> Optional[List[TEdge]]:
        """
        Восстановить путь до вершины.
        
        Args:
            v: Индекс целевой вершины
            
        Returns:
            Список ребер пути или None если пути нет
        """
        if not self.has_path_to(v):
            return None
        
        path = []
        edge = self._edges[v]
        
        while edge is not None:
            path.append(edge)
            edge = self._edges[edge.start_v]
        
        path.reverse()
        return path
