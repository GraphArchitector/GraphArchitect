"""
Адаптер GraphArchitect → LangChain.

Позволяет использовать инструменты GraphArchitect как LangChain Tools.
"""

import sys
from pathlib import Path
from typing import Optional, Type, Any, Dict
import logging

# Добавляем путь к GraphArchitect
grapharchitect_path = Path(__file__).parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

logger = logging.getLogger(__name__)

try:
    from langchain.tools import BaseTool as LangChainBaseTool
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from pydantic import Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not installed. Install: pip install langchain")

try:
    from grapharchitect.entities.base_tool import BaseTool as GraphArchitectTool
    GRAPHARCHITECT_AVAILABLE = True
except ImportError:
    GRAPHARCHITECT_AVAILABLE = False
    logger.error("GraphArchitect not available")


class GraphArchitectToolWrapper(LangChainBaseTool):
    """
    Обертка GraphArchitect инструмента для использования в LangChain.
    
    Преобразует GraphArchitect BaseTool в LangChain Tool.
    """
    
    name: str = Field(description="Название инструмента")
    description: str = Field(description="Описание возможностей")
    grapharchitect_tool: Any = Field(description="Оригинальный GraphArchitect инструмент")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, grapharchitect_tool: 'GraphArchitectTool', **kwargs):
        """
        Инициализация обертки.
        
        Args:
            grapharchitect_tool: Инструмент GraphArchitect для обертки
        """
        # Формируем описание для LangChain
        name = grapharchitect_tool.metadata.tool_name
        
        description = grapharchitect_tool.metadata.description or f"Tool: {name}"
        description += f"\nInput: {grapharchitect_tool.input.format}"
        description += f"\nOutput: {grapharchitect_tool.output.format}"
        description += f"\nReputation: {grapharchitect_tool.metadata.reputation:.2f}"
        
        super().__init__(
            name=name,
            description=description,
            grapharchitect_tool=grapharchitect_tool,
            **kwargs
        )
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Выполнить инструмент.
        
        Args:
            query: Входные данные
            run_manager: Callback manager (опционально)
            
        Returns:
            Результат выполнения
        """
        try:
            # Вызов GraphArchitect инструмента
            result = self.grapharchitect_tool.execute(query)
            
            # Логирование через callback если есть
            if run_manager:
                run_manager.on_tool_end(result)
            
            return str(result)
        
        except Exception as e:
            logger.error(f"Error executing GraphArchitect tool: {e}")
            return f"Error: {str(e)}"
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async выполнение (fallback на sync)."""
        return self._run(query, run_manager)


def convert_grapharchitect_tools_to_langchain(
    grapharchitect_tools: list
) -> list:
    """
    Конвертировать список GraphArchitect инструментов в LangChain Tools.
    
    Args:
        grapharchitect_tools: Список BaseTool из GraphArchitect
        
    Returns:
        Список LangChain Tools
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not installed")
    
    langchain_tools = []
    
    for tool in grapharchitect_tools:
        try:
            wrapped_tool = GraphArchitectToolWrapper(grapharchitect_tool=tool)
            langchain_tools.append(wrapped_tool)
        except Exception as e:
            logger.error(f"Failed to wrap tool {tool.metadata.tool_name}: {e}")
    
    logger.info(f"Converted {len(langchain_tools)} GraphArchitect tools to LangChain")
    return langchain_tools


def create_langchain_agent_with_grapharchitect_tools(
    grapharchitect_tools: list,
    llm: Any,
    agent_type: str = "zero-shot-react-description"
):
    """
    Создать LangChain агента с инструментами из GraphArchitect.
    
    Args:
        grapharchitect_tools: Список GraphArchitect инструментов
        llm: LangChain LLM (например, ChatOpenAI)
        agent_type: Тип агента LangChain
        
    Returns:
        Инициализированный LangChain агент
    """
    from langchain.agents import initialize_agent, AgentType
    
    # Конвертируем инструменты
    langchain_tools = convert_grapharchitect_tools_to_langchain(grapharchitect_tools)
    
    # Создаем агента
    agent = initialize_agent(
        tools=langchain_tools,
        llm=llm,
        agent=agent_type,
        verbose=True,
        handle_parsing_errors=True
    )
    
    logger.info(f"Created LangChain agent with {len(langchain_tools)} tools")
    return agent
