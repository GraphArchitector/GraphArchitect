"""
Протоколы для интероперабельности.

Поддержка стандартных протоколов взаимодействия агентов:
- A2A (Agent2Agent) от Google
- MCP (Model Context Protocol) от Anthropic
"""

from .a2a_protocol import (
    AgentCard,
    A2ATask,
    TaskStatus,
    MessageRole,
    A2AClient,
    A2AServer
)

from .mcp_protocol import (
    MCPTool,
    MCPToolCall,
    MCPToolResult,
    MCPServer,
    MCPClient
)

from .a2a_to_grapharchitect import (
    A2AAgentWrapper,
    wrap_a2a_agent_as_tool
)

from .mcp_to_grapharchitect import (
    MCPToolWrapper,
    convert_mcp_tools_to_grapharchitect
)

__all__ = [
    # A2A Protocol
    "AgentCard",
    "A2ATask",
    "TaskStatus",
    "MessageRole",
    "A2AClient",
    "A2AServer",
    
    # MCP Protocol
    "MCPTool",
    "MCPToolCall",
    "MCPToolResult",
    "MCPServer",
    "MCPClient",
    
    # Adapters
    "A2AAgentWrapper",
    "wrap_a2a_agent_as_tool",
    "MCPToolWrapper",
    "convert_mcp_tools_to_grapharchitect"
]
