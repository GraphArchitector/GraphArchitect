"""Ребро графа с группой инструментов"""

from typing import List, Optional
from ..algorithms.graph.base_edge import BaseEdge
from ..entities.base_tool import BaseTool
from .selection.instrument_selector import InstrumentSelector, InstrumentSelectionResult


class ToolEdge(BaseEdge):
    """
    Ребро графа с группой инструментов (ранее AgentEdge).
    
    На ребре находятся все инструменты, которые могут преобразовать
    данные из формата коннектора-истока в формат коннектора-стока.
    
    Вес ребра = средний вес всех инструментов в группе.
    """
    
    def __init__(
        self,
        start_v: int = 0,
        end_v: int = 0,
        tool: Optional[BaseTool] = None
    ):
        super().__init__(start_v, end_v)
        self.tools: List[BaseTool] = []
        
        if tool is not None:
            self.add_tool(tool)
    
    def add_tool(self, tool: BaseTool):
        """
        Добавить инструмент в группу.
        
        Args:
            tool: Инструмент для добавления
        """
        self.tools.append(tool)
        self._recalculate_weight()
    
    def _recalculate_weight(self):
        """Пересчитать вес ребра как среднее по всем инструментам"""
        if not self.tools:
            self.w = float('inf')
            return
        
        total_weight = sum(tool.get_graph_weight() for tool in self.tools)
        self.w = total_weight / len(self.tools)
    
    def select_instrument(
        self,
        selector: InstrumentSelector,
        task_embedding: Optional[List[float]],
        top_k: int = 5
    ) -> InstrumentSelectionResult:
        """
        Выбрать инструмент из группы с помощью InstrumentSelector.
        
        Args:
            selector: Селектор инструментов
            task_embedding: Эмбеддинг задачи
            top_k: Количество лучших кандидатов
            
        Returns:
            Результат выбора инструмента
        """
        return selector.select_instrument(self.tools, task_embedding, top_k)
    
    def get_group_logits(
        self,
        task_embedding: Optional[List[float]]
    ) -> dict:
        """
        Получить логиты всех инструментов в группе.
        
        Args:
            task_embedding: Эмбеддинг задачи
            
        Returns:
            Словарь {инструмент: логит}
        """
        logits = {}
        for tool in self.tools:
            logits[tool] = tool.get_logit(task_embedding)
        return logits
    
    def __repr__(self):
        return f"ToolEdge({self.start_v} -> {self.end_v}, tools={len(self.tools)}, w={self.w:.2f})"
