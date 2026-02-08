"""
Интеграция GraphArchitect с LangChain.

Позволяет использовать:
- GraphArchitect tools в LangChain
- LangChain tools в GraphArchitect
- Гибридное выполнение с объединенными возможностями
"""

from .grapharchitect_to_langchain import (
    GraphArchitectToolWrapper,
    convert_grapharchitect_tools_to_langchain,
    create_langchain_agent_with_grapharchitect_tools
)

from .langchain_to_grapharchitect import (
    LangChainToolWrapper,
    LangChainChainWrapper,
    convert_langchain_tools_to_grapharchitect
)

from .hybrid_executor import HybridExecutor

__version__ = "1.0.0"

__all__ = [
    # GraphArchitect → LangChain
    "GraphArchitectToolWrapper",
    "convert_grapharchitect_tools_to_langchain",
    "create_langchain_agent_with_grapharchitect_tools",
    
    # LangChain → GraphArchitect
    "LangChainToolWrapper",
    "LangChainChainWrapper",
    "convert_langchain_tools_to_grapharchitect",
    
    # Hybrid
    "HybridExecutor"
]
