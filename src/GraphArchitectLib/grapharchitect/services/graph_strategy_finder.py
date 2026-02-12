"""Поиск стратегий решения задачи в графе"""

from typing import List, Optional, Callable

from ..algorithms.graph.graph import GraphW
from ..algorithms.graph.shortest_path_tree import ShortestPathTree
from ..algorithms.pathfinding.dijkstra import Dijkstra
from ..algorithms.pathfinding.astar import AStar
from ..algorithms.pathfinding.yen import YenKShortestPaths
from ..algorithms.pathfinding.ant_colony import AntColonyOptimization
from ..entities.base_tool import BaseTool
from .tool_edge import ToolEdge
from .graph_builder import GraphBuilder
from .tool_expander import ToolExpander
from .pathfinding_algorithm import PathfindingAlgorithm


class GraphStrategyFinder:
    """
    Поиск стратегий решения задачи в графе инструментов.
    
    Стратегия = последовательность групп инструментов (путь в графе)
    от входного коннектора (исток) к выходному (сток).
    """
    
    def __init__(self):
        self._expander = ToolExpander()
        self._builder = GraphBuilder()
        self.astar_heuristic: Optional[Callable[[int, int], float]] = None
    
    def find_strategies(
        self,
        tools: List[BaseTool],
        start_format: str,
        end_format: str,
        limit: int = 1,
        algorithm: Optional[PathfindingAlgorithm] = None
    ) -> List[List[BaseTool]]:
        """
        Найти стратегии решения задачи.
        
        Args:
            tools: Список доступных инструментов
            start_format: Формат входного коннектора
            end_format: Формат выходного коннектора
            limit: Количество стратегий для поиска
            algorithm: Алгоритм поиска (если None - автовыбор)
            
        Returns:
            Список стратегий (каждая стратегия = список инструментов)
        """
        # Автовыбор алгоритма
        if algorithm is None:
            algorithm = (PathfindingAlgorithm.DIJKSTRA 
                        if limit == 1 
                        else PathfindingAlgorithm.YEN)
        
        # Подготовка: расширение инструментов с Any->Any семантикой
        start_sem = self._extract_semantic(start_format)
        end_sem = self._extract_semantic(end_format)
        expanded_tools = self._expander.expand(tools, start_sem, end_sem)
        
        # Построение графа
        graph = self._builder.build(expanded_tools)
        start_id = self._builder.get_node_id(start_format)
        end_id = self._builder.get_node_id(end_format)
        
        if start_id is None or end_id is None:
            return []
        
        # Поиск в зависимости от алгоритма
        if algorithm == PathfindingAlgorithm.DIJKSTRA:
            return self._find_with_dijkstra(graph, start_id, end_id)
        elif algorithm == PathfindingAlgorithm.ASTAR:
            return self._find_with_astar(graph, start_id, end_id)
        elif algorithm == PathfindingAlgorithm.YEN:
            return self._find_with_yen(graph, start_id, end_id, limit)
        elif algorithm == PathfindingAlgorithm.ANT_COLONY:
            return self._find_with_ant_colony(graph, start_id, end_id, limit)
        else:
            return []
    
    def _find_with_dijkstra(
        self,
        graph: GraphW[ToolEdge],
        start: int,
        end: int
    ) -> List[List[BaseTool]]:
        """Поиск с помощью алгоритма Дейкстры"""
        result = []
        dijkstra = Dijkstra(graph, start)
        spt = ShortestPathTree(dijkstra.edges, dijkstra.distances)
        path = spt.get_path(end)
        
        if path and len(path) > 0:
            result.append(self._convert_path(path))
        
        return result
    
    def _find_with_astar(
        self,
        graph: GraphW[ToolEdge],
        start: int,
        end: int
    ) -> List[List[BaseTool]]:
        """Поиск с помощью A*"""
        result = []
        astar = AStar(graph, start, end, self.astar_heuristic)
        path = astar.get_path()
        
        if path and len(path) > 0:
            result.append(self._convert_path(path))
        
        return result
    
    def _find_with_yen(
        self,
        graph: GraphW[ToolEdge],
        start: int,
        end: int,
        limit: int
    ) -> List[List[BaseTool]]:
        """Поиск с помощью алгоритма Йена"""
        result = []
        yen = YenKShortestPaths(graph, start, end)
        paths = yen.get_paths(limit)
        
        for path in paths:
            result.append(self._convert_path(path))
        
        return result
    
    def _find_with_ant_colony(
        self,
        graph: GraphW[ToolEdge],
        start: int,
        end: int,
        limit: int
    ) -> List[List[BaseTool]]:
        """Поиск с помощью муравьиного алгоритма"""
        result = []
        aco = AntColonyOptimization(
            graph, start, end,
            num_ants=20,
            num_iterations=100
        )
        paths = aco.get_paths(limit)
        
        for path in paths:
            result.append(self._convert_path(path))
        
        return result
    
    def _convert_path(self, edges: List[ToolEdge]) -> List[BaseTool]:
        """
        Преобразовать путь (последовательность ребер) в стратегию.
        
        Из каждой группы инструментов на ребре выбираем лучший
        (с наивысшей репутацией).
        
        Args:
            edges: Список ребер пути
            
        Returns:
            Список инструментов (стратегия)
        """
        strategy = []
        for edge in edges:
            best_tool = max(
                edge.tools,
                key=lambda t: t.metadata.reputation
            )
            strategy.append(best_tool)
        return strategy
    
    def _extract_semantic(self, format_str: str) -> str:
        """
        Извлечь семантическую часть из формата.
        
        Args:
            format_str: Формат вида "data|semantic"
            
        Returns:
            Семантическая часть
        """
        if '|' in format_str:
            return format_str.split('|')[1]
        return ""
    
    def get_graph_builder(self) -> GraphBuilder:
        """Получить построитель графа"""
        return self._builder
