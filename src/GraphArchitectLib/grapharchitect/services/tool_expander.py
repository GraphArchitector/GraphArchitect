"""Расширитель инструментов для Any->Any семантики"""

from typing import List, Set
from ..entities.base_tool import BaseTool
from ..entities.connectors.connector import ANY_SEMANTIC


class ToolExpander:
    """
    Расширитель инструментов.
    
    Создает дополнительные копии инструментов с Any (*) семантикой,
    подставляя конкретные семантики из контекста задачи.
    
    Правило проброса: * на входе и выходе = ОДНА И ТА ЖЕ семантика
    проходит насквозь (pass-through). Инструмент меняет data_format,
    но сохраняет семантику.
    """
    
    def expand(
        self,
        tools: List[BaseTool],
        start_semantic: str,
        end_semantic: str
    ) -> List[BaseTool]:
        """
        Расширить список инструментов.
        
        Для инструментов с Any семантикой создаются копии:
        - * на входе: подставляем start_semantic
        - * на выходе: подставляем end_semantic
        - * на входе И выходе: проброс (pass-through) — одна и та же семантика
          на входе и выходе. Создаем клоны для start, end и всех уникальных
          семантик из других инструментов.
        
        Args:
            tools: Исходный список инструментов
            start_semantic: Семантика входа задачи
            end_semantic: Семантика выхода задачи
            
        Returns:
            Расширенный список инструментов
        """
        # Собираем все уникальные семантики из конкретных инструментов
        all_semantics: Set[str] = set()
        if start_semantic:
            all_semantics.add(start_semantic)
        if end_semantic:
            all_semantics.add(end_semantic)
        
        for tool in tools:
            if tool.input.semantic_format != ANY_SEMANTIC:
                all_semantics.add(tool.input.semantic_format)
            if tool.output.semantic_format != ANY_SEMANTIC:
                all_semantics.add(tool.output.semantic_format)
        
        expanded = list(tools)  # Копируем исходные
        
        for tool in tools:
            input_sem = tool.input.semantic_format
            output_sem = tool.output.semantic_format
            
            if input_sem == ANY_SEMANTIC and output_sem == ANY_SEMANTIC:
                # Оба * — проброс (pass-through):
                # Создаем клоны с ОДИНАКОВОЙ семантикой на входе и выходе
                for sem in all_semantics:
                    clone = tool.clone()
                    clone.input.input_semantic = sem
                    clone.output.input_semantic = sem
                    expanded.append(clone)
            
            elif input_sem == ANY_SEMANTIC:
                # Только вход * — подставляем start_semantic
                if start_semantic:
                    clone = tool.clone()
                    clone.input.input_semantic = start_semantic
                    expanded.append(clone)
                # Также для всех известных семантик (для промежуточных путей)
                for sem in all_semantics:
                    if sem != start_semantic:
                        clone = tool.clone()
                        clone.input.input_semantic = sem
                        expanded.append(clone)
            
            elif output_sem == ANY_SEMANTIC:
                # Только выход * — подставляем end_semantic
                if end_semantic:
                    clone = tool.clone()
                    clone.output.input_semantic = end_semantic
                    expanded.append(clone)
                # Также для всех известных семантик
                for sem in all_semantics:
                    if sem != end_semantic:
                        clone = tool.clone()
                        clone.output.input_semantic = sem
                        expanded.append(clone)
        
        return expanded
