"""Построитель графа инструментов"""

from typing import List, Dict, Optional

from ..algorithms.graph.graph import GraphW
from ..entities.base_tool import BaseTool
from .tool_edge import ToolEdge


class GraphBuilder:
    """
    Построитель графа инструментов.
    
    Граф строится следующим образом:
    - Вершины = уникальные форматы коннекторов (строки вида "data|semantic")
    - Ребра = группы инструментов, преобразующих из одного формата в другой
    - Вес ребра = среднее качество преобразования по всем инструментам
    """
    
    def __init__(self):
        self._vertex_map: Dict[str, int] = {} 
    
    def build(self, tools: List[BaseTool]) -> GraphW[ToolEdge]:
        """
        Построить граф из списка инструментов.
        
        Args:
            tools: Список инструментов
            
        Returns:
            Взвешенный граф с ребрами типа ToolEdge
        """
        self._vertex_map.clear()
        edges: Dict[str, ToolEdge] = {}
        id_counter = 0
        
        # Проходим по всем инструментам
        for tool in tools:
            start_fmt = tool.input.format
            end_fmt = tool.output.format
            
            # Добавляем вершины если их еще нет
            if start_fmt not in self._vertex_map:
                self._vertex_map[start_fmt] = id_counter
                id_counter += 1
            
            if end_fmt not in self._vertex_map:
                self._vertex_map[end_fmt] = id_counter
                id_counter += 1
            
            # Получаем ID вершин
            u = self._vertex_map[start_fmt]
            v = self._vertex_map[end_fmt]
            key = f"{u}_{v}"
            
            # Создаем или дополняем ребро
            if key not in edges:
                edges[key] = ToolEdge(u, v, tool)
            else:
                edges[key].add_tool(tool)
        
        # Создаем граф
        graph = GraphW[ToolEdge](id_counter)
        for edge in edges.values():
            graph.add_edge(edge)
        
        return graph
    
    def get_node_id(self, format_str: str) -> Optional[int]:
        """
        Получить ID вершины по формату.
        
        Args:
            format_str: Формат коннектора
            
        Returns:
            ID вершины или None если формат не найден
        """
        return self._vertex_map.get(format_str)
    
    def get_format_by_id(self, node_id: int) -> Optional[str]:
        """
        Получить формат по ID вершины.
        
        Args:
            node_id: ID вершины
            
        Returns:
            Формат коннектора или None если ID не найден
        """
        for fmt, vid in self._vertex_map.items():
            if vid == node_id:
                return fmt
        return None
    
    @property
    def vertex_map(self) -> Dict[str, int]:
        """Маппинг формат -> ID вершины"""
        return self._vertex_map.copy()
