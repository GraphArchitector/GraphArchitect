"""
Адаптер MCP → GraphArchitect.

Позволяет использовать MCP серверы как инструменты GraphArchitect.
"""

import logging
from typing import Optional, Dict, Any

from ..entities.base_tool import BaseTool
from ..entities.connectors.connector import Connector
from .mcp_protocol import MCPClient, MCPTool

logger = logging.getLogger(__name__)


class MCPToolWrapper(BaseTool):
    """
    Обертка MCP инструмента для GraphArchitect.
    
    Преобразует MCP Tool в GraphArchitect BaseTool.
    """
    
    def __init__(
        self,
        mcp_client: MCPClient,
        mcp_tool: MCPTool,
        input_connector: Optional[Connector] = None,
        output_connector: Optional[Connector] = None
    ):
        """
        Инициализация обертки.
        
        Args:
            mcp_client: MCP клиент для вызовов
            mcp_tool: Описание MCP инструмента
            input_connector: Входной коннектор (дефолт: text|question)
            output_connector: Выходной коннектор (дефолт: text|answer)
        """
        super().__init__()
        
        self._mcp_client = mcp_client
        self._mcp_tool = mcp_tool
        
        # Метаданные
        self.metadata.tool_name = mcp_tool.name
        self.metadata.description = mcp_tool.description
        self.metadata.reputation = 0.80  # Начальная для MCP tools
        
        # Коннекторы
        self.input = input_connector or Connector("text", "question")
        self.output = output_connector or Connector("text", "answer")
    
    def execute(self, input_data: Any) -> str:
        """
        Выполнить MCP инструмент.
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат выполнения
        """
        try:
            # Формируем аргументы для MCP
            arguments = {"input": str(input_data)}
            
            # Вызов через MCP клиент
            result = self._mcp_client.call_tool(
                self._mcp_tool.name,
                arguments
            )
            
            if result is not None:
                return str(result)
            else:
                return f"[MCP Error] Failed to execute {self._mcp_tool.name}"
        
        except Exception as e:
            logger.error(f"Error executing MCP tool: {e}")
            return f"[MCP Error] {str(e)}"


def convert_mcp_tools_to_grapharchitect(
    mcp_client: MCPClient,
    connector_mappings: Optional[Dict[str, tuple]] = None
) -> list:
    """
    Конвертировать MCP инструменты в GraphArchitect BaseTool.
    
    Args:
        mcp_client: MCP клиент с обнаруженными инструментами
        connector_mappings: Маппинг названий на коннекторы
            {"tool_name": (input_connector, output_connector)}
            
    Returns:
        Список GraphArchitect BaseTool
    """
    connector_mappings = connector_mappings or {}
    tools = []
    
    for mcp_tool in mcp_client.get_available_tools():
        # Получаем коннекторы из маппинга
        connectors = connector_mappings.get(mcp_tool.name)
        
        if connectors:
            input_conn, output_conn = connectors
        else:
            input_conn, output_conn = None, None
        
        # Создаем обертку
        wrapped_tool = MCPToolWrapper(
            mcp_client=mcp_client,
            mcp_tool=mcp_tool,
            input_connector=input_conn,
            output_connector=output_conn
        )
        
        tools.append(wrapped_tool)
    
    logger.info(f"Converted {len(tools)} MCP tools to GraphArchitect")
    return tools
