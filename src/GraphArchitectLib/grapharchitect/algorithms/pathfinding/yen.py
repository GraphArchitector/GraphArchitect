"""Алгоритм Йена для поиска K кратчайших путей"""

from typing import List, TypeVar, Generic
import heapq
import copy

from ..graph.base_edge import BaseEdge
from ..graph.graph import GraphW
from ..graph.shortest_path_tree import ShortestPathTree
from .dijkstra import Dijkstra

TEdge = TypeVar('TEdge', bound=BaseEdge)


class YenKShortestPaths(Generic[TEdge]):
    """
    Алгоритм Йена для поиска K кратчайших путей.
    
    Находит топ-K простых путей (без циклов) между двумя вершинами.
    """
    
    def __init__(self, graph: GraphW[TEdge], start: int, end: int):
        """
        Инициализация алгоритма Йена.
        
        Args:
            graph: Граф
            start: Начальная вершина
            end: Конечная вершина
        """
        self._graph = graph
        self._start = start
        self._end = end
    
    def get_paths(self, k: int) -> List[List[TEdge]]:
        """
        Найти K кратчайших путей.
        
        Args:
            k: Количество путей для поиска
            
        Returns:
            Список из не более чем K путей
        """
        # A - список найденных путей
        a: List[List[TEdge]] = []
        
        # B - очередь кандидатов
        b: List[tuple[float, List[TEdge]]] = []
        
        # Найти первый кратчайший путь
        dijkstra = Dijkstra(self._graph, self._start)
        spt = ShortestPathTree(dijkstra.edges, dijkstra.distances)
        
        initial_path = spt.get_path(self._end)
        if initial_path is None or len(initial_path) == 0:
            return a
        
        a.append(initial_path)
        
        # Ищем k-1 дополнительных путей
        for k_curr in range(1, k):
            prev_path = a[k_curr - 1]
            
            for i in range(len(prev_path)):
                spur_node = prev_path[i].start_v
                root_path = prev_path[:i]
                
                # Временно удаляем ребра
                removed_edges: List[TEdge] = []
                
                for p in a:
                    if len(p) > i and self._paths_equal(root_path, p[:i]):
                        edge_to_remove = p[i]
                        try:
                            self._graph.remove_edge(edge_to_remove)
                            removed_edges.append(edge_to_remove)
                        except:
                            pass
                
                # Ищем путь от spur_node до end
                spur_dijkstra = Dijkstra(self._graph, spur_node)
                spur_spt = ShortestPathTree(spur_dijkstra.edges, spur_dijkstra.distances)
                spur_path = spur_spt.get_path(self._end)
                
                if spur_path and len(spur_path) > 0:
                    total_path = root_path + spur_path
                    cost = sum(e.w for e in total_path)
                    heapq.heappush(b, (cost, total_path))
                
                # Восстанавливаем удаленные ребра
                for edge in removed_edges:
                    self._graph.add_edge(edge)
            
            if not b:
                break
            
            # Берем лучший кандидат
            while b:
                best_cost, best_path = heapq.heappop(b)
                
                # Проверяем на дубликаты
                is_duplicate = False
                for p in a:
                    if self._paths_equal(p, best_path):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    a.append(best_path)
                    break
        
        return a
    
    def _paths_equal(self, p1: List[TEdge], p2: List[TEdge]) -> bool:
        """Проверка равенства двух путей"""
        if len(p1) != len(p2):
            return False
        
        for e1, e2 in zip(p1, p2):
            if e1 != e2:
                return False
        
        return True
