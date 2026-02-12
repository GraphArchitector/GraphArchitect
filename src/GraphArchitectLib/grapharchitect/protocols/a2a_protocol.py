"""
Базовая поддержка протокола Agent2Agent (A2A) от Google.

Реализует:
- Agent Card формат
- Task lifecycle
- Message exchange
- Basic A2A client/server
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Статусы задачи в A2A протоколе."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(Enum):
    """Роли сообщений в A2A."""
    USER = "user"
    AGENT = "agent"


@dataclass
class AgentCard:
    """
    Agent Card - JSON описание агента для A2A.
    
    Публикуется по пути /.well-known/agent.json
    """
    
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoint: str  # URL для взаимодействия
    
    version: str = "1.0.0"
    supported_formats: List[str] = field(default_factory=list)
    authentication: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в dict для JSON."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "version": self.version,
            "supported_formats": self.supported_formats,
            "authentication": self.authentication,
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AgentCard':
        """Десериализация из dict."""
        return AgentCard(
            agent_id=data['agent_id'],
            name=data['name'],
            description=data['description'],
            capabilities=data['capabilities'],
            endpoint=data['endpoint'],
            version=data.get('version', '1.0.0'),
            supported_formats=data.get('supported_formats', []),
            authentication=data.get('authentication', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class A2ATask:
    """
    Задача в A2A протоколе.
    
    Основная единица работы между агентами.
    """
    
    task_id: str
    user_message: str
    status: TaskStatus = TaskStatus.PENDING
    
    context: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    artifact: Optional[Any] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация."""
        return {
            "task_id": self.task_id,
            "user_message": self.user_message,
            "status": self.status.value,
            "context": self.context,
            "messages": self.messages,
            "artifact": self.artifact,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_agent": self.assigned_agent
        }


class A2AClient:
    """
    Клиент для взаимодействия с A2A агентами.
    
    Позволяет GraphArchitect инструментам общаться через A2A.
    """
    
    def __init__(self, agent_card_url: str):
        """
        Инициализация клиента.
        
        Args:
            agent_card_url: URL к /.well-known/agent.json
        """
        self._agent_card_url = agent_card_url
        self._agent_card: Optional[AgentCard] = None
        
        logger.info(f"A2A Client initialized for: {agent_card_url}")
    
    def discover_agent(self) -> Optional[AgentCard]:
        """
        Обнаружить агента через Agent Card.
        
        Returns:
            AgentCard если найден
        """
        try:
            import requests
            
            response = requests.get(self._agent_card_url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            self._agent_card = AgentCard.from_dict(data)
            
            logger.info(f"Discovered agent: {self._agent_card.name}")
            return self._agent_card
        
        except Exception as e:
            logger.error(f"Failed to discover agent: {e}")
            return None
    
    def create_task(self, user_message: str, context: Dict = None) -> Optional[A2ATask]:
        """
        Создать задачу для агента.
        
        Args:
            user_message: Сообщение пользователя
            context: Дополнительный контекст
            
        Returns:
            Созданная задача
        """
        if not self._agent_card:
            logger.error("Agent not discovered. Call discover_agent() first")
            return None
        
        task = A2ATask(
            task_id=str(uuid.uuid4()),
            user_message=user_message,
            context=context or {}
        )
        
        try:
            import requests
            
            response = requests.post(
                f"{self._agent_card.endpoint}/tasks",
                json=task.to_dict(),
                timeout=10
            )
            
            response.raise_for_status()
            
            result_data = response.json()
            task.status = TaskStatus(result_data.get('status', 'pending'))
            task.assigned_agent = self._agent_card.agent_id
            
            logger.info(f"Task created: {task.task_id}")
            return task
        
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Получить статус задачи.
        
        Args:
            task_id: ID задачи
            
        Returns:
            Статус задачи
        """
        if not self._agent_card:
            return None
        
        try:
            import requests
            
            response = requests.get(
                f"{self._agent_card.endpoint}/tasks/{task_id}",
                timeout=5
            )
            
            response.raise_for_status()
            data = response.json()
            
            return TaskStatus(data.get('status', 'pending'))
        
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return None


class A2AServer:
    """
    Базовый A2A сервер.
    
    Позволяет GraphArchitect инструментам быть доступными через A2A.
    """
    
    def __init__(self, agent_card: AgentCard):
        """
        Инициализация сервера.
        
        Args:
            agent_card: Карточка этого агента
        """
        self._agent_card = agent_card
        self._tasks: Dict[str, A2ATask] = {}
        
        logger.info(f"A2A Server initialized: {agent_card.name}")
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Получить Agent Card для публикации."""
        return self._agent_card.to_dict()
    
    def create_task(self, user_message: str, context: Dict = None) -> A2ATask:
        """Создать новую задачу."""
        task = A2ATask(
            task_id=str(uuid.uuid4()),
            user_message=user_message,
            context=context or {},
            assigned_agent=self._agent_card.agent_id
        )
        
        self._tasks[task.task_id] = task
        logger.info(f"Task created: {task.task_id}")
        
        return task
    
    def get_task(self, task_id: str) -> Optional[A2ATask]:
        """Получить задачу по ID."""
        return self._tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: TaskStatus):
        """Обновить статус задачи."""
        task = self._tasks.get(task_id)
        if task:
            task.status = status
            task.updated_at = datetime.now()
