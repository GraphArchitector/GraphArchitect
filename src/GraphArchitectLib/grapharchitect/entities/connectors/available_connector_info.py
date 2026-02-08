"""Агрегированная информация о доступных коннекторах"""

from typing import Set
from dataclasses import dataclass, field


@dataclass
class AvailableConnectorInfo:
    """
    Агрегированная информация о доступных коннекторах от всех инструментов.
    
    Используется в ЕЯИ для фильтрации возможных вариантов.
    """
    
    # Типы входных файлов: txt, csv, json, png, jpg, pdf, mp3, wav и т.д.
    file_types: Set[str] = field(default_factory=set)
    
    # Типы внутренних представлений (complex_type):
    # matrix, vector, tensor, signal, text, image, sound
    complex_types: Set[str] = field(default_factory=set)
    
    # Уточняющие подтипы:
    # raw, specter, cepstrum, speak, question и т.д.
    subtypes: Set[str] = field(default_factory=set)
    
    # Семантические типы входа:
    # specter, cepstrum, speak, question и т.д.
    semantic_input_types: Set[str] = field(default_factory=set)
    
    # Семантические типы выхода:
    # answer, summary, report и т.д.
    semantic_output_types: Set[str] = field(default_factory=set)
    
    # Доступные доменные области:
    # physics, linguistics, medicine, statistics и т.д.
    knowledge_domains: Set[str] = field(default_factory=set)
