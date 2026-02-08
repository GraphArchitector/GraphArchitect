"""Коннектор - формат данных (вершина в графе)"""

from typing import Optional
from dataclasses import dataclass


# Константы
ANY_SEMANTIC = "*"


@dataclass
class Connector:
    """
    Коннектор определяет формат данных.
    
    Формат коннектора = DataFormat|SemanticFormat
    Например: "text|question", "matrix|specter", "file.json|raw"
    """
    
    data_format: str = ""
    semantic_format: str = ANY_SEMANTIC
    
    # Входная семантика (для проброса контекста в Any->Any инструментах)
    input_semantic: Optional[str] = None
    
    @property
    def format(self) -> str:
        """
        Полный формат коннектора.
        
        Returns:
            Строка формата "data|semantic"
        """
        return self.get_format(
            self.semantic_format, 
            self.data_format, 
            self.input_semantic
        )
    
    @staticmethod
    def get_format(
        semantic_format: str, 
        data_format: str, 
        input_semantic: Optional[str] = None
    ) -> str:
        """
        Построить формат коннектора.
        
        Args:
            semantic_format: Семантический формат
            data_format: Формат данных
            input_semantic: Входная семантика (опционально)
            
        Returns:
            Строка формата
        """
        s_format = input_semantic if (
            input_semantic and semantic_format == ANY_SEMANTIC
        ) else semantic_format
        
        return f"{data_format}|{s_format}"
    
    def clone(self):
        """Клонировать коннектор"""
        import copy
        return copy.deepcopy(self)
