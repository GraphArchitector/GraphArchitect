"""Агрегатор информации о доступных коннекторах"""

from typing import List

from ...entities.base_tool import BaseTool
from ...entities.connectors.available_connector_info import AvailableConnectorInfo


class ConnectorInfoAggregator:
    """
    Агрегатор информации о доступных коннекторах.
    
    Собирает информацию из всех инструментов о том, какие
    типы данных и форматы доступны в системе.
    """
    
    def aggregate_from_tools(
        self,
        tools: List[BaseTool]
    ) -> AvailableConnectorInfo:
        """
        Агрегировать информацию о коннекторах из инструментов.
        
        Args:
            tools: Список инструментов
            
        Returns:
            Агрегированная информация о коннекторах
        """
        info = AvailableConnectorInfo()
        
        for tool in tools:
            # Обрабатываем входной коннектор
            self._process_connector(
                tool.input.data_format,
                tool.input.semantic_format,
                info,
                is_input=True
            )
            
            # Обрабатываем выходной коннектор
            self._process_connector(
                tool.output.data_format,
                tool.output.semantic_format,
                info,
                is_input=False
            )
        
        return info
    
    def _process_connector(
        self,
        data_format: str,
        semantic_format: str,
        info: AvailableConnectorInfo,
        is_input: bool
    ):
        """
        Обработать один коннектор.
        
        Args:
            data_format: Формат данных
            semantic_format: Семантический формат
            info: Агрегированная информация
            is_input: True если входной коннектор
        """
        # Определяем тип (file или structured)
        if data_format.startswith("file."):
            # Это файл
            file_ext = data_format.replace("file.", "")
            info.file_types.add(file_ext)
        else:
            # Это structured тип
            info.complex_types.add(data_format)
            info.subtypes.add(data_format)
        
        # Добавляем семантический тип
        if semantic_format and semantic_format != "*":
            if is_input:
                info.semantic_input_types.add(semantic_format)
            else:
                info.semantic_output_types.add(semantic_format)
