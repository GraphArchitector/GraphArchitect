"""Тип данных коннектора"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class DataType:
    """
    Тип данных коннектора.
    
    Описывает физический формат данных.
    """
    
    # Сложный тип: "file" или "structured"
    complex_type: str = ""
    
    # Подтип:
    # - для file: расширение (txt, json, pdf, jpg, mp3, wav)
    # - для structured: тип (matrix, vector, tensor, text, image, sound, signal)
    subtype: str = ""
    
    # Дополнительные характеристики
    characteristics: Dict[str, Any] = field(default_factory=dict)
