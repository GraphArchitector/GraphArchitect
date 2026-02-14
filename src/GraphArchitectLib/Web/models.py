"""
Data models for API.
"""
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============== Инструменты ==============

class Agent(BaseModel):
    """Tool in processing chain."""
    id: str
    name: str
    icon: str
    color: str
    type: str = "general"
    specialization: Optional[str] = None
    capabilities: List[str] = []
    cost: float = 0.0  # Cost per operation
    metrics: Dict[str, Any] = {}


class CandidateProgress(BaseModel):
    """Progress of candidate in competitive selection."""
    agent_id: str
    status: Literal["competing", "leading", "eliminated", "winner"] = "competing"
    progress: int = 0  # 0-100
    score: Optional[float] = None  # 0.0-1.0


class SelectionCriteria(BaseModel):
    """Tool selection criteria."""
    strategy: Literal["fastest_response", "best_quality_score", "consensus", "balanced"] = "best_quality_score"
    timeout: int = 10000  # milliseconds


class WorkflowStep(BaseModel):
    """Workflow step (1 of N tools is selected within the step)."""
    id: str
    name: str
    order: int
    description: Optional[str] = None
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    phase: Optional[Literal["selection", "executing", "completed"]] = None
    
    # Инструменты-кандидаты для этого шага (выбирается 1 из N)
    candidate_agents: List[str] = Field(default_factory=list, alias="candidateAgents")
    
    # Выбранный агент (после конкурентного отбора)
    selected_agent_id: Optional[str] = Field(default=None, alias="selectedAgentId")
    
    # Критерии выбора агента
    selection_criteria: SelectionCriteria = Field(default_factory=SelectionCriteria, alias="selectionCriteria")
    
    # Прогресс отбора кандидатов (для real-time обновлений)
    candidates_progress: List[CandidateProgress] = Field(default_factory=list, alias="candidatesProgress")
    
    # Результат выполнения
    result: Optional[Dict[str, Any]] = None
    
    class Config:
        populate_by_name = True


class WorkflowChain(BaseModel):
    """Цепочка шагов для обработки (шаги выполняются последовательно)"""
    chat_id: str
    name: str = "Default Workflow"
    description: Optional[str] = None
    
    # Шаги выполняются последовательно
    steps: List[WorkflowStep] = []
    
    # Индекс текущего шага
    current_step_index: int = Field(default=0, alias="currentStepIndex")
    
    created_at: datetime = Field(default_factory=datetime.now)
    request_type: Literal["text", "image", "combined"] = "text"
    
    # Файлы, прикрепленные к workflow
    files: List[str] = []
    
    # Старый формат для обратной совместимости
    agents: List[Agent] = []
    
    class Config:
        populate_by_name = True


# ============== Модели сообщений ==============

class MessageRequest(BaseModel):
    """Запрос на отправку сообщения"""
    chat_id: str
    message: str
    files: Optional[List[str]] = []
    planning_algorithm: str = "yen_5"
    use_streaming: bool = True
    use_rewoo: bool = False
    user_priority: str = "balanced"
    max_cost: Optional[float] = None
    max_time: Optional[float] = None

class MessageChunk(BaseModel):
    """Чанк ответа (для стриминга всей цепочки: Генерация -> Выбор -> Выполнение)"""
    type: Literal[
        "gen_phase_start", "gen_progress", "gen_phase_complete", 
        "step_started", "agent_progress", "agent_score_updated", 
        "agent_selected", "agent_executing", "step_completed",
        "workflow_info", "text", "error"
    ]
    content: Optional[str] = None
    phase_id: Optional[str] = None
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    progress: Optional[int] = None
    score: Optional[float] = None
    metadata: Optional[dict] = None


class MessageResponse(BaseModel):
    """Полный ответ на сообщение"""
    chat_id: str
    message: str
    response_type: Literal["text", "image", "document", "combined"]
    response_data: Union[str, dict]
    workflow_used: List[Agent]
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.now)


# ============== Модели документов ==============

class DocumentUpload(BaseModel):
    """Загрузка документа"""
    chat_id: str
    filename: str
    content_type: str
    size: int


class DocumentInfo(BaseModel):
    """Информация о сохраненном документе"""
    document_id: str
    chat_id: str
    filename: str
    content_type: str
    size: int
    path: str
    uploaded_at: datetime = Field(default_factory=datetime.now)


# ============== Модели создания workflow ==============

class WorkflowCreateRequest(BaseModel):
    """Запрос на создание цепочки агентов"""
    chat_id: str
    request_type: str = "text"
    user_message: str
    files: Optional[List[str]] = []
    planning_algorithm: str = "yen_5"
    use_streaming: bool = True

class WorkflowCreateResponse(BaseModel):
    """Ответ с созданной цепочкой"""
    chat_id: str
    workflow: WorkflowChain
    message: str = "Workflow created successfully"


# ============== Модели чата ==============

class ChatInfo(BaseModel):
    """Информация о чате"""
    chat_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    workflow_chain: Optional[WorkflowChain] = None
    documents: List[DocumentInfo] = []


# ============== Модели ответа API ==============

class ApiResponse(BaseModel):
    """Стандартный ответ API"""
    success: bool
    message: str
    data: Optional[dict] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Ответ с ошибкой"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
