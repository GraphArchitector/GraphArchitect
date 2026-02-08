"""
Адаптер LangChain → GraphArchitect.

Позволяет использовать LangChain chains и tools как инструменты GraphArchitect.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any
import logging

# Добавляем путь к GraphArchitect
grapharchitect_path = Path(__file__).parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

logger = logging.getLogger(__name__)

try:
    from grapharchitect.entities.base_tool import BaseTool
    from grapharchitect.entities.connectors.connector import Connector
    GRAPHARCHITECT_AVAILABLE = True
except ImportError:
    GRAPHARCHITECT_AVAILABLE = False
    logger.error("GraphArchitect not available")

try:
    from langchain.tools import BaseTool as LangChainTool
    from langchain.chains.base import Chain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not installed. Install: pip install langchain")


class LangChainToolWrapper(BaseTool):
    """
    Обертка LangChain Tool для использования в GraphArchitect.
    
    Преобразует LangChain Tool в GraphArchitect BaseTool.
    """
    
    def __init__(
        self,
        langchain_tool: 'LangChainTool',
        input_connector: Optional[Connector] = None,
        output_connector: Optional[Connector] = None
    ):
        """
        Инициализация обертки.
        
        Args:
            langchain_tool: LangChain инструмент
            input_connector: Входной коннектор (если None, используется text|question)
            output_connector: Выходной коннектор (если None, используется text|answer)
        """
        super().__init__()
        
        self._langchain_tool = langchain_tool
        
        # Метаданные из LangChain tool
        self.metadata.tool_name = langchain_tool.name
        self.metadata.description = langchain_tool.description or ""
        self.metadata.reputation = 0.75  # Начальная репутация
        
        # Коннекторы (по умолчанию text|question → text|answer)
        self.input = input_connector or Connector("text", "question")
        self.output = output_connector or Connector("text", "answer")
    
    def execute(self, input_data: Any) -> str:
        """
        Выполнить LangChain инструмент.
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат выполнения
        """
        try:
            # Вызов LangChain tool
            result = self._langchain_tool.run(str(input_data))
            return str(result)
        
        except Exception as e:
            logger.error(f"Error executing LangChain tool: {e}")
            return f"Error: {str(e)}"


class LangChainChainWrapper(BaseTool):
    """
    Обертка LangChain Chain для использования в GraphArchitect.
    
    Преобразует LangChain Chain в GraphArchitect BaseTool.
    """
    
    def __init__(
        self,
        chain: 'Chain',
        name: str,
        description: str = "",
        input_connector: Optional[Connector] = None,
        output_connector: Optional[Connector] = None,
        input_key: str = "input",
        output_key: str = "output"
    ):
        """
        Инициализация обертки.
        
        Args:
            chain: LangChain Chain
            name: Название инструмента
            description: Описание
            input_connector: Входной коннектор
            output_connector: Выходной коннектор
            input_key: Ключ для входных данных в chain
            output_key: Ключ для выходных данных из chain
        """
        super().__init__()
        
        self._chain = chain
        self._input_key = input_key
        self._output_key = output_key
        
        # Метаданные
        self.metadata.tool_name = name
        self.metadata.description = description
        self.metadata.reputation = 0.80  # Chains обычно надежнее
        
        # Коннекторы
        self.input = input_connector or Connector("text", "question")
        self.output = output_connector or Connector("text", "answer")
    
    def execute(self, input_data: Any) -> str:
        """
        Выполнить LangChain Chain.
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат выполнения
        """
        try:
            # Вызов chain
            result = self._chain.run({self._input_key: str(input_data)})
            
            # Извлечение результата
            if isinstance(result, dict) and self._output_key in result:
                return str(result[self._output_key])
            else:
                return str(result)
        
        except Exception as e:
            logger.error(f"Error executing LangChain chain: {e}")
            return f"Error: {str(e)}"


def convert_langchain_tools_to_grapharchitect(
    langchain_tools: list,
    connector_mappings: Optional[Dict[str, tuple]] = None
) -> list:
    """
    Конвертировать LangChain tools в GraphArchitect BaseTool.
    
    Args:
        langchain_tools: Список LangChain Tools
        connector_mappings: Маппинг названий на коннекторы
            {"tool_name": (input_connector, output_connector)}
            
    Returns:
        Список GraphArchitect BaseTool
    """
    if not GRAPHARCHITECT_AVAILABLE:
        raise ImportError("GraphArchitect not available")
    
    connector_mappings = connector_mappings or {}
    grapharchitect_tools = []
    
    for tool in langchain_tools:
        # Получаем коннекторы из маппинга или используем дефолтные
        connectors = connector_mappings.get(tool.name)
        
        if connectors:
            input_conn, output_conn = connectors
        else:
            input_conn, output_conn = None, None
        
        try:
            wrapped_tool = LangChainToolWrapper(
                langchain_tool=tool,
                input_connector=input_conn,
                output_connector=output_conn
            )
            grapharchitect_tools.append(wrapped_tool)
        except Exception as e:
            logger.error(f"Failed to wrap LangChain tool {tool.name}: {e}")
    
    logger.info(f"Converted {len(grapharchitect_tools)} LangChain tools to GraphArchitect")
    return grapharchitect_tools
