"""Определение задачи - входные данные для системы"""

from typing import Optional, List, Any
from datetime import datetime
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from .connectors.connector import Connector


@dataclass
class TaskDefinition:
    """
    Определение задачи без ЕЯИ - прямое указание коннекторов.
    
    После обработки ЕЯИ текстовое описание преобразуется в пару коннекторов:
    входной (исток) и выходной (сток).
    """
    
    task_id: UUID = field(default_factory=uuid4)
    input_connector: Connector = field(default_factory=Connector)
    output_connector: Connector = field(default_factory=Connector)
    description: str = ""
    task_embedding: Optional[List[float]] = None
    input_data: Optional[Any] = None
    domain: str = "general"  # Область знаний: physics, linguistics, medicine и т.д.
    created_at: datetime = field(default_factory=datetime.utcnow)
