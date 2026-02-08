"""Описание коннектора для ЕЯИ"""

from typing import Optional
from dataclasses import dataclass

from .data_type import DataType
from .semantic_type import SemanticType


@dataclass
class ConnectorDescriptor:
    """
    Описание коннектора в естественно-языковом интерфейсе (ЕЯИ).
    
    Используется для преобразования текстового описания задачи
    в конкретные коннекторы.
    """
    
    data_type: Optional[DataType] = None
    semantic_type: Optional[SemanticType] = None
    
    # Область знаний: physics, linguistics, medicine, statistics и т.д.
    knowledge_domain: str = ""
