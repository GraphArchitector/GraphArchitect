"""
Гибридный исполнитель: GraphArchitect + LangChain.

Объединяет сильные стороны обеих систем:
- GraphArchitect: планирование графа, softmax выбор, обучение
- LangChain: богатая экосистема tools, chains, agents
"""

import sys
from pathlib import Path
from typing import List, Optional, Any, Dict
import logging

# Добавляем путь к GraphArchitect
grapharchitect_path = Path(__file__).parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

logger = logging.getLogger(__name__)

from grapharchitect.entities.task_definition import TaskDefinition
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.services.execution.execution_orchestrator import ExecutionOrchestrator
from grapharchitect.services.execution.execution_context import ExecutionContext
from grapharchitect.services.selection.instrument_selector import InstrumentSelector
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm

from grapharchitect_to_langchain import convert_grapharchitect_tools_to_langchain
from langchain_to_grapharchitect import convert_langchain_tools_to_grapharchitect


class HybridExecutor:
    """
    Гибридный исполнитель задач.
    
    Использует:
    - GraphArchitect для планирования и выбора
    - LangChain tools в качестве исполнителей
    - Policy Gradient для обучения
    """
    
    def __init__(
        self,
        embedding_service=None,
        temperature_constant: float = 1.0,
        learning_rate: float = 0.01
    ):
        """
        Инициализация гибридного исполнителя.
        
        Args:
            embedding_service: Сервис эмбеддингов
            temperature_constant: Константа температуры для softmax
            learning_rate: Скорость обучения
        """
        # Сервисы GraphArchitect
        self.embedding_service = embedding_service or SimpleEmbeddingService(dimension=384)
        self.selector = InstrumentSelector(temperature_constant=temperature_constant)
        self.strategy_finder = GraphStrategyFinder()
        self.orchestrator = ExecutionOrchestrator(
            self.embedding_service,
            self.selector,
            self.strategy_finder
        )
        
        # Списки инструментов
        self.grapharchitect_tools = []
        self.langchain_tools = []
        self.all_tools = []
        
        logger.info("HybridExecutor initialized")
    
    def add_grapharchitect_tools(self, tools: List):
        """
        Добавить GraphArchitect инструменты.
        
        Args:
            tools: Список GraphArchitect BaseTool
        """
        self.grapharchitect_tools.extend(tools)
        self.all_tools.extend(tools)
        
        logger.info(f"Added {len(tools)} GraphArchitect tools")
    
    def add_langchain_tools(
        self,
        tools: List,
        connector_mappings: Optional[Dict[str, tuple]] = None
    ):
        """
        Добавить LangChain инструменты (конвертируются в GraphArchitect).
        
        Args:
            tools: Список LangChain Tools
            connector_mappings: Маппинг на коннекторы
        """
        wrapped_tools = convert_langchain_tools_to_grapharchitect(
            tools,
            connector_mappings
        )
        
        self.langchain_tools.extend(tools)
        self.all_tools.extend(wrapped_tools)
        
        logger.info(f"Added {len(tools)} LangChain tools")
    
    def execute_task(
        self,
        description: str,
        input_data: str,
        input_connector: Optional[Connector] = None,
        output_connector: Optional[Connector] = None,
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.YEN,
        path_limit: int = 5,
        top_k: int = 5
    ) -> ExecutionContext:
        """
        Выполнить задачу используя гибридный набор инструментов.
        
        Args:
            description: Описание задачи
            input_data: Входные данные
            input_connector: Входной коннектор (или auto-detect)
            output_connector: Выходной коннектор (или auto-detect)
            algorithm: Алгоритм поиска путей
            path_limit: Лимит путей
            top_k: Топ-K для softmax
            
        Returns:
            Контекст выполнения с результатами
        """
        # Коннекторы по умолчанию
        input_connector = input_connector or Connector("text", "question")
        output_connector = output_connector or Connector("text", "answer")
        
        # Создание задачи
        task = TaskDefinition(
            description=description,
            input_connector=input_connector,
            output_connector=output_connector,
            input_data=input_data
        )
        
        # Выполнение через GraphArchitect Orchestrator
        # Он использует ВСЕ инструменты (и GraphArchitect, и LangChain)
        context = self.orchestrator.execute_task(
            task=task,
            available_tools=self.all_tools,
            path_limit=path_limit,
            top_k=top_k
        )
        
        logger.info(f"Hybrid execution completed: {context.status.value}")
        return context
    
    def get_langchain_agent(self, llm: Any):
        """
        Получить LangChain агента со всеми инструментами.
        
        Args:
            llm: LangChain LLM
            
        Returns:
            LangChain Agent
        """
        # Конвертируем GraphArchitect tools в LangChain
        langchain_wrapped = convert_grapharchitect_tools_to_langchain(
            self.grapharchitect_tools
        )
        
        # Объединяем с нативными LangChain tools
        all_langchain_tools = langchain_wrapped + self.langchain_tools
        
        # Создаем агента
        from langchain.agents import initialize_agent
        
        agent = initialize_agent(
            tools=all_langchain_tools,
            llm=llm,
            agent="zero-shot-react-description",
            verbose=True
        )
        
        return agent
    
    def get_tools_summary(self) -> Dict[str, Any]:
        """
        Получить сводку по инструментам.
        
        Returns:
            Словарь с информацией об инструментах
        """
        return {
            "grapharchitect_tools": len(self.grapharchitect_tools),
            "langchain_tools": len(self.langchain_tools),
            "total_tools": len(self.all_tools),
            "tools": [
                {
                    "name": tool.metadata.tool_name,
                    "input": tool.input.format,
                    "output": tool.output.format,
                    "reputation": tool.metadata.reputation,
                    "source": "grapharchitect" if tool in self.grapharchitect_tools else "langchain"
                }
                for tool in self.all_tools
            ]
        }
