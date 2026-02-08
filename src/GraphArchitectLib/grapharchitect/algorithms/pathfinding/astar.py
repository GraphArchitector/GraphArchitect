"""Алгоритм A* для поиска кратчайшего пути с эвристикой"""

from typing import List, Optional, TypeVar, Generic, Callable
import heapq

from ..graph.base_edge import BaseEdge
from ..graph.graph import GraphW

TEdge = TypeVar('TEdge', bound=BaseEdge)


class AStar(Generic[TEdge]):
    """
    Алгоритм A* для поиска кратчайшего пути с эвристикой.
    
    Если эвристика h(v, target) = 0, то A* эквивалентен Дейкстре.
    """
    
    def __init__(
        self,
        graph: GraphW[TEdge],
        start: int,
        target: int,
        heuristic: Optional[Callable[[int, int], float]] = None
    ):
        """
        Инициализация и выполнение алгоритма A*.
        
        Args:
            graph: Граф
            start: Начальная вершина
            target: Целевая вершина
            heuristic: Эвристическая функция h(v, target).
                      Если None, работает как Дейкстра
        """
        self._graph = graph
        self._start = start
        self._target = target
        self._heuristic = heuristic if heuristic else lambda v, t: 0.0
        
        # Массивы расстояний и ребер
        self._distances: List[float] = [float('inf')] * graph.v
        self._edge_to: List[Optional[TEdge]] = [None] * graph.v
        
        # Выполнить алгоритм
        self._search()
    
    def _search(self):
        """Выполнить поиск кратчайшего пути"""
        # Priority queue: (f_score, vertex) где f_score = g_score + h_score
        pq = [(self._heuristic(self._start, self._target), self._start)]
        self._distances[self._start] = 0.0
        visited = set()
        
        while pq:
            _, v = heapq.heappop(pq)
            
            # Если достигли цели - можем остановиться
            if v == self._target:
                break
            
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
                    
                    # f_score = g_score + h_score
                    f_score = new_distance + self._heuristic(w, self._target)
                    heapq.heappush(pq, (f_score, w))
    
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
    
    def get_path(self) -> Optional[List[TEdge]]:
        """
        Получить путь до целевой вершины.
        
        Returns:
            Список ребер пути или None если пути нет
        """
        if not self.has_path_to(self._target):
            return None
        
        path = []
        edge = self._edge_to[self._target]
        
        while edge is not None:
            path.append(edge)
            edge = self._edge_to[edge.start_v]
        
        path.reverse()
        return path
