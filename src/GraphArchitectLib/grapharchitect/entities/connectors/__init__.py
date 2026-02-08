"""Модуль коннекторов - определение форматов данных"""

from .connector import Connector
from .connector_descriptor import ConnectorDescriptor
from .data_type import DataType
from .semantic_type import SemanticType
from .available_connector_info import AvailableConnectorInfo
from .task_representation import TaskRepresentation

__all__ = [
    'Connector',
    'ConnectorDescriptor',
    'DataType',
    'SemanticType',
    'AvailableConnectorInfo',
    'TaskRepresentation'
]
