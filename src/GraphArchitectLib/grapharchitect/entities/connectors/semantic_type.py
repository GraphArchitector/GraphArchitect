"""Семантический тип данных"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class SemanticType:
    """
    Семантический тип данных.
    
    Описывает смысловое содержание данных.
    """
    
    # Категория семантического содержания:
    # raw, specter, cepstrum, speak, question, answer, summary, reasoning, report
    semantic_category: str = ""
    
    # Характеристики категории
    characteristics: Dict[str, Any] = field(default_factory=dict)
