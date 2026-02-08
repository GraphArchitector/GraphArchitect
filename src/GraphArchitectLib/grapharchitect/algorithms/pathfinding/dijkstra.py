"""Алгоритм Дейкстры для поиска кратчайшего пути"""

from typing import List, Optional, TypeVar, Generic
import heapq

from ..graph.base_edge import BaseEdge
from ..graph.graph import GraphW

TEdge = TypeVar('TEdge', bound=BaseEdge)


class Dijkstra(Generic[TEdge]):
    """
    Алгоритм Дейкстры для поиска кратчайшего пути
    от одной вершины до всех остальных.
    """
    
    def __init__(self, graph: GraphW[TEdge], start: int):
        """
        Инициализация и выполнение алгоритма Дейкстры.
        
        Args:
            graph: Граф
            start: Начальная вершина
        """
        self._graph = graph
        self._start = start
        
        # Массивы расстояний и ребер
        self._distances: List[float] = [float('inf')] * graph.v
        self._edge_to: List[Optional[TEdge]] = [None] * graph.v
        
        # Выполнить алгоритм
        self._search()
    
    def _search(self):
        """Выполнить поиск кратчайших путей"""
        # Priority queue: (distance, vertex)
        pq = [(0.0, self._start)]
        self._distances[self._start] = 0.0
        visited = set()
        
        while pq:
            dist, v = heapq.heappop(pq)
            
            if v in visited:
                continue
            
            visited.add(v)
            
            # Релаксация всех соседей
            for edge in self._graph.adj(v):
                w = edge.end_v
                
                if w in visited:
                    continue
                
                new_distance = self._distances[v] + edge.w
                
                if new_distance < self._distances[w]:
                    self._distances[w] = new_distance
                    self._edge_to[w] = edge
                    heapq.heappush(pq, (new_distance, w))
    
    @property
    def distances(self) -> List[float]:
        """Массив расстояний до всех вершин"""
        return self._distances
    
    @property
    def edges(self) -> List[Optional[TEdge]]:
        """Массив ребер дерева кратчайших путей"""
        return self._edge_to
    
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
