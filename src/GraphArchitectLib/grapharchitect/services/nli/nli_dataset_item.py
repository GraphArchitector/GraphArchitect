"""Элемент датасета для обучения ЕЯИ"""

from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from ...entities.connectors.task_representation import TaskRepresentation


@dataclass
class NLIDatasetItem:
    """
    Элемент датасета для обучения естественно-языкового интерфейса.
    
    Содержит:
    - Доступные типы данных и коннекторов в системе
    - Текст задачи на естественном языке
    - Правильные входной и выходной коннекторы
    """
    
    # Доступные типы входных файлов в системе
    file_types: List[str] = field(default_factory=list)
    
    # Доступные сложные типы данных (complex types)
    complex_types: List[str] = field(default_factory=list)
    
    # Доступные подтипы
    subtypes: List[str] = field(default_factory=list)
    
    # Доступные семантические типы на входе
    semantic_input_types: List[str] = field(default_factory=list)
    
    # Доступные семантические типы на выходе
    semantic_output_types: List[str] = field(default_factory=list)
    
    # Доступные доменные области знаний
    knowledge_domains: List[str] = field(default_factory=list)
    
    # Текст задачи на естественном языке
    task_text: str = ""
    
    # Векторное представление текста задачи
    task_embedding: Optional[List[float]] = None
    
    # JSON-представление задачи (входной и выходной коннекторы)
    representation: Optional[TaskRepresentation] = None
    
    # Метка времени создания
    created_at: datetime = field(default_factory=datetime.utcnow)
