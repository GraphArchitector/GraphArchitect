"""
Адаптер A2A → GraphArchitect.

Позволяет использовать A2A агентов как инструменты GraphArchitect.
"""

import logging
from typing import Optional
import time

from ..entities.base_tool import BaseTool
from ..entities.connectors.connector import Connector
from .a2a_protocol import A2AClient, AgentCard, TaskStatus

logger = logging.getLogger(__name__)


class A2AAgentWrapper(BaseTool):
    """
    Обертка A2A агента для GraphArchitect.
    
    Преобразует A2A Agent в GraphArchitect BaseTool.
    """
    
    def __init__(
        self,
        a2a_client: A2AClient,
        agent_card: AgentCard,
        input_connector: Optional[Connector] = None,
        output_connector: Optional[Connector] = None,
        timeout: int = 60
    ):
        """
        Инициализация обертки.
        
        Args:
            a2a_client: A2A клиент
            agent_card: Карточка агента
            input_connector: Входной коннектор
            output_connector: Выходной коннектор
            timeout: Таймаут ожидания выполнения (секунды)
        """
        super().__init__()
        
        self._a2a_client = a2a_client
        self._agent_card = agent_card
        self._timeout = timeout
        
        # Метаданные из Agent Card
        self.metadata.tool_name = agent_card.name
        self.metadata.description = agent_card.description
        self.metadata.reputation = 0.85  # Начальная для A2A agents
        
        # Коннекторы
        self.input = input_connector or Connector("text", "question")
        self.output = output_connector or Connector("text", "answer")
    
    def execute(self, input_data) -> str:
        """
        Выполнить через A2A агента.
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат выполнения
        """
        try:
            # Создаем задачу
            task = self._a2a_client.create_task(
                user_message=str(input_data)
            )
            
            if not task:
                return f"[A2A Error] Failed to create task"
            
            # Ожидание выполнения (polling)
            elapsed = 0
            poll_interval = 1  # секунда
            
            while elapsed < self._timeout:
                status = self._a2a_client.get_task_status(task.task_id)
                
                if status == TaskStatus.COMPLETED:
                    # Задача выполнена
                    # В реальности нужно получить результат через API
                    return f"[A2A] Task completed by {self._agent_card.name}"
                
                elif status == TaskStatus.FAILED:
                    return f"[A2A Error] Task failed"
                
                elif status == TaskStatus.CANCELLED:
                    return f"[A2A Error] Task cancelled"
                
                # Ожидание
                time.sleep(poll_interval)
                elapsed += poll_interval
            
            # Таймаут
            logger.warning(f"A2A task timeout after {self._timeout}s")
            return f"[A2A Error] Timeout after {self._timeout}s"
        
        except Exception as e:
            logger.error(f"Error executing A2A agent: {e}")
            return f"[A2A Error] {str(e)}"


def wrap_a2a_agent_as_tool(
    agent_card_url: str,
    input_connector: Optional[Connector] = None,
    output_connector: Optional[Connector] = None
) -> Optional[BaseTool]:
    """
    Обернуть A2A агента как GraphArchitect инструмент.
    
    Args:
        agent_card_url: URL к /.well-known/agent.json
        input_connector: Входной коннектор
        output_connector: Выходной коннектор
        
    Returns:
        GraphArchitect BaseTool или None
    """
    # Создаем клиент
    client = A2AClient(agent_card_url)
    
    # Обнаруживаем агента
    agent_card = client.discover_agent()
    
    if not agent_card:
        logger.error(f"Failed to discover A2A agent at {agent_card_url}")
        return None
    
    # Создаем обертку
    wrapped_tool = A2AAgentWrapper(
        a2a_client=client,
        agent_card=agent_card,
        input_connector=input_connector,
        output_connector=output_connector
    )
    
    logger.info(f"Wrapped A2A agent: {agent_card.name}")
    return wrapped_tool
