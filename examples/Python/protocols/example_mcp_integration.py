"""
Пример интеграции с MCP (Model Context Protocol).

Демонстрирует:
- Создание MCP сервера с GraphArchitect инструментами
- Использование MCP инструментов в GraphArchitect
- JSON-RPC коммуникация
"""

import sys
from pathlib import Path

# Добавляем GraphArchitect
grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

from grapharchitect.protocols.mcp_protocol import MCPServer, MCPClient, MCPTool
from grapharchitect.protocols.mcp_to_grapharchitect import MCPToolWrapper, convert_mcp_tools_to_grapharchitect
from grapharchitect.entities.connectors.connector import Connector

print("=" * 70)
print("ПРИМЕР: Интеграция MCP с GraphArchitect")
print("=" * 70)
print()

# Шаг 1: Создание MCP сервера
print("Шаг 1: Создание MCP сервера")
print("-" * 70)

server = MCPServer(server_name="grapharchitect-demo")

# Регистрируем инструменты
def search_handler(query: str) -> str:
    """Поиск информации"""
    return f"Search results for: {query}"

def calculator_handler(expression: str) -> str:
    """Калькулятор"""
    try:
        result = eval(expression)  # DEMO только! В production использовать безопасный парсер
        return f"Result: {result}"
    except:
        return "Error: Invalid expression"

server.register_tool(
    name="search",
    description="Searches for information on the web",
    handler=search_handler,
    parameters={
        "query": {"type": "string", "description": "Search query"}
    }
)

server.register_tool(
    name="calculator",
    description="Performs mathematical calculations",
    handler=calculator_handler,
    parameters={
        "expression": {"type": "string", "description": "Math expression"}
    }
)

print(f"  [OK] MCP Server created: {server._server_name}")
print(f"  Registered tools: {len(server._tools)}")
print()

# Шаг 2: List tools (MCP метод)
print("Шаг 2: Обнаружение инструментов (list_tools)")
print("-" * 70)

tools_list = server.list_tools()

print(f"  Доступно инструментов: {len(tools_list)}")
for tool in tools_list:
    print(f"    - {tool['name']}: {tool['description']}")

print()

# Шаг 3: Call tool (MCP метод)
print("Шаг 3: Вызов инструмента (call_tool)")
print("-" * 70)

# Вызов search
result1 = server.call_tool("search", {"query": "GraphArchitect framework"})
print(f"  Search:")
print(f"    Success: {result1.success}")
print(f"    Result: {result1.result}")

# Вызов calculator
result2 = server.call_tool("calculator", {"expression": "2 + 2 * 3"})
print(f"  Calculator:")
print(f"    Success: {result2.success}")
print(f"    Result: {result2.result}")

print()

# Шаг 4: JSON-RPC обработка
print("Шаг 4: JSON-RPC запрос")
print("-" * 70)

jsonrpc_request = {
    "jsonrpc": "2.0",
    "method": "list_tools",
    "params": {},
    "id": 1
}

response = server.handle_jsonrpc_request(jsonrpc_request)

print(f"  Запрос: {jsonrpc_request['method']}")
print(f"  Ответ: {len(response['result'])} tools")
print()

# Шаг 5: Конвертация в GraphArchitect (если есть клиент)
print("Шаг 5: Использование MCP в GraphArchitect")
print("-" * 70)

print("  [INFO] Для использования MCP инструментов в GraphArchitect:")
print()
print("  1. Создать MCP Client:")
print("     client = MCPClient('http://localhost:8080')")
print()
print("  2. Обнаружить инструменты:")
print("     client.discover_tools()")
print()
print("  3. Конвертировать:")
print("     ga_tools = convert_mcp_tools_to_grapharchitect(client)")
print()
print("  4. Использовать в GraphArchitect:")
print("     orchestrator.execute_task(task, ga_tools)")
print()

# Итоги
print("=" * 70)
print("ИТОГИ")
print("=" * 70)
print()
print("MCP протокол реализован:")
print("  - MCPServer экспонирует инструменты")
print("  - list_tools() для обнаружения")
print("  - call_tool() для выполнения")
print("  - JSON-RPC 2.0 коммуникация")
print()
print("Интеграция с GraphArchitect:")
print("  - MCP инструменты можно использовать в графе")
print("  - Автоматическая конвертация в BaseTool")
print("  - Указание коннекторов через mappings")
print()
print("Применение:")
print("  - Подключение внешних сервисов (БД, Search, APIs)")
print("  - Использование community MCP servers")
print("  - Интероперабельность с другими AI системами")
print()
print("=" * 70)
