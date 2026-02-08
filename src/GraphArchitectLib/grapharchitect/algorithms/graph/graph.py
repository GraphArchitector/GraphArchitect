"""Взвешенный ориентированный граф"""

from typing import List, TypeVar, Generic

from .base_edge import BaseEdge

TEdge = TypeVar('TEdge', bound=BaseEdge)


class GraphW(Generic[TEdge]):
    """
    Взвешенный ориентированный граф с типизированными ребрами.
    
    Вершины представлены индексами (int), ребра - объектами типа TEdge.
    """
    
    def __init__(self, v: int):
        """
        Инициализация графа.
        
        Args:
            v: Количество вершин
        """
        self._v = v  # Количество вершин
        self._e = 0  # Количество ребер
        self._adj: List[List[TEdge]] = [[] for _ in range(v)]  # Списки смежности
    
    @property
    def v(self) -> int:
        """Количество вершин"""
        return self._v
    
    @property
    def e(self) -> int:
        """Количество ребер"""
        return self._e
    
    def add_edge(self, edge: TEdge):
        """
        Добавить ребро в граф.
        
        Args:
            edge: Ребро для добавления
        """
        v = edge.start_v
        if 0 <= v < self._v:
            self._adj[v].append(edge)
            self._e += 1
        else:
            raise ValueError(f"Неверный индекс вершины: {v}")
    
    def adj(self, v: int) -> List[TEdge]:
        """
        Получить список смежных ребер для вершины.
        
        Args:
            v: Индекс вершины
            
        Returns:
            Список ребер, исходящих из вершины v
        """
        if 0 <= v < self._v:
            return self._adj[v]
        else:
            raise ValueError(f"Неверный индекс вершины: {v}")
    
    def remove_edge(self, edge: TEdge):
        """
        Удалить ребро из графа.
        
        Args:
            edge: Ребро для удаления
        """
        v = edge.start_v
        if 0 <= v < self._v and edge in self._adj[v]:
            self._adj[v].remove(edge)
            self._e -= 1
    
    def edges(self) -> List[TEdge]:
        """
        Получить все ребра графа.
        
        Returns:
            Список всех ребер
        """
        all_edges = []
        for v in range(self._v):
            all_edges.extend(self._adj[v])
        return all_edges
    
    def __repr__(self):
        return f"GraphW(v={self._v}, e={self._e})"
