"""Реактивный инструмент - простая реализация BaseTool"""

from typing import Callable, Any
import sys
from pathlib import Path

# Добавляем путь к библиотеке grapharchitect
project_root = Path(__file__).parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(project_root))

from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector


class ReactiveTool(BaseTool):
    """
    Реактивный инструмент (ранее ReactiveAgent).
    
    Простая реализация инструмента с функцией-обработчиком.
    Используется для тестирования и демонстрации.
    """
    
    def __init__(
        self,
        input_data_format: str,
        input_semantic_format: str,
        output_data_format: str,
        output_semantic_format: str,
        name: str,
        execute_func: Callable[[Any], Any]
    ):
        """
        Инициализация реактивного инструмента.
        
        Args:
            input_data_format: Формат входных данных
            input_semantic_format: Семантический формат входа
            output_data_format: Формат выходных данных
            output_semantic_format: Семантический формат выхода
            name: Имя инструмента
            execute_func: Функция для выполнения
        """
        super().__init__()
        
        # Настройка коннекторов
        self.input = Connector(
            data_format=input_data_format,
            semantic_format=input_semantic_format
        )
        self.output = Connector(
            data_format=output_data_format,
            semantic_format=output_semantic_format
        )
        
        # Метаданные
        self.metadata.tool_name = name
        self.metadata.description = f"{name}: {input_data_format}|{input_semantic_format} -> {output_data_format}|{output_semantic_format}"
        
        # Функция выполнения
        self._execute_func = execute_func
    
    def execute(self, input_data: Any) -> Any:
        """
        Выполнить инструмент.
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат выполнения
        """
        return self._execute_func(input_data)
