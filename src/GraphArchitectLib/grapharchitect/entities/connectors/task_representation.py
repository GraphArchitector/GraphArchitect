"""Представление задачи для ЕЯИ"""

from typing import Optional
from dataclasses import dataclass

from .connector_descriptor import ConnectorDescriptor


@dataclass
class TaskRepresentation:
    """
    Полное представление задачи для естественно-языкового интерфейса (ЕЯИ).
    
    Результат работы ЕЯИ: текст задачи -> пара коннекторов (входной, выходной).
    """
    
    input_connector: Optional[ConnectorDescriptor] = None
    output_connector: Optional[ConnectorDescriptor] = None
