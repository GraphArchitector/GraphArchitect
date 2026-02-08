"""
Базовая поддержка Model Context Protocol (MCP) от Anthropic.

Реализует:
- MCP Server интерфейс
- Tool discovery (list_tools)
- Tool execution (call_tool)
- JSON-RPC 2.0 коммуникация
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """
    Описание инструмента в MCP формате.
    
    Возвращается методом list_tools().
    """
    
    name: str
    description: str
    
    # Параметры (JSON Schema)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для MCP."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "metadata": self.metadata
        }


@dataclass
class MCPToolCall:
    """Вызов инструмента через MCP."""
    
    tool_name: str
    arguments: Dict[str, Any]
    
    # ID для отслеживания
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MCPToolResult:
    """Результат выполнения инструмента."""
    
    call_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация."""
        return {
            "call_id": self.call_id,
            "success": self.success,
            "result": self.result,
            "error": self.error
        }


class MCPServer:
    """
    MCP Server для экспонирования GraphArchitect инструментов.
    
    Позволяет любым MCP-совместимым агентам использовать
    GraphArchitect инструменты.
    """
    
    def __init__(self, server_name: str = "grapharchitect-mcp"):
        """
        Инициализация MCP сервера.
        
        Args:
            server_name: Название сервера
        """
        self._server_name = server_name
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions: Dict[str, MCPTool] = {}
        
        logger.info(f"MCP Server initialized: {server_name}")
    
    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Dict[str, Any] = None
    ):
        """
        Зарегистрировать инструмент.
        
        Args:
            name: Название инструмента
            description: Описание возможностей
            handler: Функция-обработчик
            parameters: JSON Schema параметров
        """
        self._tools[name] = handler
        
        self._tool_descriptions[name] = MCPTool(
            name=name,
            description=description,
            parameters=parameters or {}
        )
        
        logger.info(f"MCP tool registered: {name}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        MCP метод: list_tools().
        
        Возвращает список доступных инструментов.
        
        Returns:
            Список описаний инструментов
        """
        return [tool.to_dict() for tool in self._tool_descriptions.values()]
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """
        MCP метод: call_tool().
        
        Выполняет указанный инструмент.
        
        Args:
            tool_name: Название инструмента
            arguments: Аргументы для инструмента
            
        Returns:
            Результат выполнения
        """
        call_id = str(uuid.uuid4())
        
        # Проверка существования инструмента
        if tool_name not in self._tools:
            return MCPToolResult(
                call_id=call_id,
                success=False,
                error=f"Tool not found: {tool_name}"
            )
        
        # Выполнение
        try:
            handler = self._tools[tool_name]
            result = handler(**arguments)
            
            return MCPToolResult(
                call_id=call_id,
                success=True,
                result=result
            )
        
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            
            return MCPToolResult(
                call_id=call_id,
                success=False,
                error=str(e)
            )
    
    def handle_jsonrpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработать JSON-RPC 2.0 запрос.
        
        Args:
            request: JSON-RPC запрос
            
        Returns:
            JSON-RPC ответ
        """
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')
        
        # Роутинг методов
        if method == 'list_tools':
            result = self.list_tools()
            
        elif method == 'call_tool':
            tool_name = params.get('name')
            arguments = params.get('arguments', {})
            
            tool_result = self.call_tool(tool_name, arguments)
            result = tool_result.to_dict()
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        
        # Формирование ответа
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }


class MCPClient:
    """
    MCP Client для использования удаленных инструментов.
    
    Позволяет GraphArchitect использовать MCP серверы.
    """
    
    def __init__(self, server_url: str):
        """
        Инициализация клиента.
        
        Args:
            server_url: URL MCP сервера
        """
        self._server_url = server_url.rstrip('/')
        self._available_tools: List[MCPTool] = []
        
        logger.info(f"MCP Client initialized for: {server_url}")
    
    def discover_tools(self) -> List[MCPTool]:
        """
        Обнаружить доступные инструменты (list_tools).
        
        Returns:
            Список доступных инструментов
        """
        try:
            import requests
            
            # JSON-RPC запрос
            request = {
                "jsonrpc": "2.0",
                "method": "list_tools",
                "params": {},
                "id": 1
            }
            
            response = requests.post(
                self._server_url,
                json=request,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            if 'result' in data:
                tools_data = data['result']
                self._available_tools = [
                    MCPTool(
                        name=t['name'],
                        description=t['description'],
                        parameters=t.get('parameters', {}),
                        metadata=t.get('metadata', {})
                    )
                    for t in tools_data
                ]
                
                logger.info(f"Discovered {len(self._available_tools)} MCP tools")
                return self._available_tools
            
            else:
                logger.error(f"No result in response: {data}")
                return []
        
        except Exception as e:
            logger.error(f"Failed to discover MCP tools: {e}")
            return []
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """
        Вызвать инструмент на MCP сервере.
        
        Args:
            tool_name: Название инструмента
            arguments: Аргументы
            
        Returns:
            Результат выполнения или None
        """
        try:
            import requests
            
            # JSON-RPC запрос
            request = {
                "jsonrpc": "2.0",
                "method": "call_tool",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                },
                "id": 2
            }
            
            response = requests.post(
                self._server_url,
                json=request,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if 'result' in data:
                result = data['result']
                
                if result.get('success'):
                    return result.get('result')
                else:
                    logger.error(f"Tool execution failed: {result.get('error')}")
                    return None
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to call MCP tool: {e}")
            return None
    
    def get_available_tools(self) -> List[MCPTool]:
        """Получить список доступных инструментов."""
        return self._available_tools
