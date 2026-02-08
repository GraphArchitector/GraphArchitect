"""
Bridge between Web API and GraphArchitect library.

This module integrates GraphArchitect functionality into Web API:
- Agent to BaseTool conversion
- Real graph strategy search
- Tool selection via softmax with temperature
- Execution streaming with gradient traces
"""

import sys
import logging
from pathlib import Path

# Add path to grapharchitect
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Optional, AsyncGenerator, Tuple, Dict, Any
import asyncio
import uuid

logger = logging.getLogger(__name__)

from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.connectors.connector import Connector, ANY_SEMANTIC
from grapharchitect.entities.task_definition import TaskDefinition
from grapharchitect.services.execution.execution_orchestrator import ExecutionOrchestrator
from grapharchitect.services.execution.execution_context import ExecutionContext
from grapharchitect.services.selection.instrument_selector import InstrumentSelector
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm
from grapharchitect.services.nli.natural_language_interface import NaturalLanguageInterface
from grapharchitect.services.nli.nli_dataset_item import NLIDatasetItem
from grapharchitect.services.training.training_orchestrator import TrainingOrchestrator
from grapharchitect.services.feedback.feedback_data import FeedbackData, FeedbackSource
from grapharchitect.services.feedback.simple_critic import SimpleCritic

# ReWOO Planning (опционально)
try:
    from grapharchitect.planning.rewoo_planner import ReWOOPlanner
    REWOO_AVAILABLE = True
except ImportError:
    REWOO_AVAILABLE = False
    logger.warning("ReWOO planner not available")

from models import Agent, MessageChunk
from repository import get_repository


class AgentTool(BaseTool):
    """
    Адаптер Agent → BaseTool.
    
    Преобразует Agent модель из Web API в BaseTool для использования
    в GraphArchitect. Автоматически выводит коннекторы на основе типа агента.
    """
    
    def __init__(self, agent: Agent):
        super().__init__()
        
        # Копируем метаданные из Agent
        self.metadata.tool_name = agent.name
        self.metadata.description = agent.specialization or ""
        self.metadata.reputation = agent.metrics.get("avgScore", 0.85)
        self.metadata.mean_cost = agent.cost
        self.metadata.mean_time_answer = agent.metrics.get("avgResponseTime", 3000) / 1000
        
        # Инициализация статистики для обучения
        self.metadata.training_sample_size = 10  # Начальное значение
        self.metadata.variance_estimate = 0.1
        
        # Определяем коннекторы на основе типа агента
        self.input, self.output = self._infer_connectors(agent)
        
        # Сохраняем ссылку на оригинальный Agent
        self._agent = agent
        self._agent_id = agent.id
    
    def _infer_connectors(self, agent: Agent) -> Tuple[Connector, Connector]:
        """
        Вывести коннекторы из типа агента.
        
        Маппинг основан на семантике операций, которые выполняет агент.
        """
        connector_mappings = {
            "classification": (
                Connector("text", "question"),
                Connector("text", "category")
            ),
            "content_generation": (
                Connector("text", "outline"),
                Connector("text", "content")
            ),
            "quality_assurance": (
                Connector("text", "content"),
                Connector("text", "validated")
            ),
            "research": (
                Connector("text", "query"),
                Connector("text", "findings")
            ),
            "planning": (
                Connector("text", "topic"),
                Connector("text", "outline")
            ),
            "writing": (
                Connector("text", "outline"),
                Connector("text", "article")
            ),
            "editing": (
                Connector("text", "draft"),
                Connector("text", "polished")
            ),
            "code_analysis": (
                Connector("text", "code"),
                Connector("text", "analysis")
            ),
            "reporting": (
                Connector("text", "data"),
                Connector("text", "report")
            ),
            "image_generation": (
                Connector("text", ANY_SEMANTIC),
                Connector("image", ANY_SEMANTIC)
            ),
            "image_processing": (
                Connector("image", "raw"),
                Connector("text", "description")
            ),
            "text_extraction": (
                Connector("image", "raw"),
                Connector("text", "extracted")
            ),
            # Дополнительные маппинги для новых типов
            "parsing": (
                Connector("text", "raw"),
                Connector("text", "parsed")
            ),
            "analysis": (
                Connector("text", "question"),
                Connector("text", "analysis")
            ),
            "qa": (
                Connector("text", "question"),
                Connector("text", "answer")
            ),
            "universal": (
                Connector("text", "question"),
                Connector("text", "answer")
            ),
        }
        
        # Получаем маппинг или используем универсальный question→answer
        return connector_mappings.get(
            agent.type,
            (Connector("text", "question"), Connector("text", "answer"))
        )
    
    def execute(self, input_data):
        """
        Выполнить агента.
        
        При наличии OPENROUTER_API_KEY: реальный LLM/image вызов.
        Без ключа: заглушка.
        """
        import os
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if api_key:
            # Генерация изображений — отдельная ветка
            if self._agent.type == "image_generation":
                return self._execute_image_generation(input_data, api_key)
            
            try:
                from grapharchitect.tools.ApiTools.OpenRouterTool.openrouter_llm import OpenRouterLLM
                
                # Системный промпт: роль инструмента
                system_prompts = {
                    "classification": "Ты классификатор текста. Определи категорию/тональность сообщения. Отвечай кратко на русском.",
                    "content_generation": "Ты генератор контента. Создай текст по запросу пользователя. Отвечай на русском.",
                    "quality_assurance": "Ты контролер качества. Проверь текст и укажи замечания. Отвечай на русском.",
                    "research": "Ты исследователь. Дай подробный ответ по теме. Отвечай на русском.",
                    "qa": "Ты помощник. Ответь на вопрос пользователя точно и полезно. Отвечай на русском.",
                    "universal": "Ты умный помощник. Обработай запрос и дай полезный ответ. Отвечай на русском.",
                    "analysis": "Ты аналитик. Проанализируй данные и дай выводы. Отвечай на русском.",
                    "planning": "Ты планировщик. Составь план по запросу. Отвечай на русском.",
                    "writing": "Ты писатель. Напиши текст по запросу. Отвечай на русском.",
                    "editing": "Ты редактор. Улучши текст. Отвечай на русском.",
                    "reporting": "Ты составитель отчетов. Создай отчет. Отвечай на русском.",
                }
                
                # Выбор модели по типу агента и репутации
                model_by_agent = {
                    # Высокая репутация → Claude Sonnet 4.5
                    "classification": ("anthropic/claude-sonnet-4.5", 0.98),
                    "qa": ("anthropic/claude-sonnet-4.5", 0.92),
                    
                    # Выше среднего → Gemini 3 Flash
                    "content_generation": ("google/gemini-3-flash-preview", 0.85),
                    "writing": ("google/gemini-3-flash-preview", 0.90),
                    "research": ("google/gemini-3-flash-preview", 0.86),
                    
                    # Средняя → Gemini 2.5 Flash
                    "universal": ("google/gemini-2.5-flash", 0.80),
                    "analysis": ("google/gemini-2.5-flash", 0.88),
                    "planning": ("google/gemini-2.5-flash", 0.89),
                    
                    # Низкая → Gemini 2.5 Flash Lite
                    "quality_assurance": ("google/gemini-2.5-flash-lite", 0.76),
                    "editing": ("google/gemini-2.5-flash-lite", 0.87),
                    "reporting": ("google/gemini-2.5-flash-lite", 0.84),
                }
                
                model_name, target_rep = model_by_agent.get(
                    self._agent.type,
                    ("google/gemini-2.5-flash", 0.80)
                )
                
                # Обновляем репутацию агента под модель
                self.metadata.reputation = target_rep
                
                system_prompt = system_prompts.get(
                    self._agent.type, 
                    "Ты умный помощник. Ответь на запрос пользователя полезно и на русском языке."
                )
                
                llm = OpenRouterLLM(
                    api_key=api_key,
                    model_name=model_name,
                    system_prompt=system_prompt
                )
                
                user_input = str(input_data).strip()
                
                result = llm.query_llm(
                    question=user_input,
                    temperature=0.7,
                    max_tokens=4000
                )
                
                logger.info(f"OpenRouter executed: {self.metadata.tool_name}")
                return result
            
            except Exception as e:
                logger.warning(f"OpenRouter error for {self.metadata.tool_name}: {e}")
        
        # Fallback
        return f"[{self.metadata.tool_name}] Processed: {str(input_data)[:100]}"
    
    def _execute_image_generation(self, input_data, api_key: str) -> str:
        """
        Генерация изображения через OpenRouter.
        
        Формат ответа: message.images[].image_url.url (base64 data URL).
        """
        import requests
        
        user_input = str(input_data).strip()
        
        # Модели с поддержкой image output
        image_models = [
            {"model": "google/gemini-2.5-flash-image", "modalities": ["image", "text"]},
        ]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/FractalAgentsAI/GraphArchitect",
            "X-Title": "GraphArchitect"
        }
        
        for model_cfg in image_models:
            try:
                payload = {
                    "model": model_cfg["model"],
                    "messages": [
                        {"role": "user", "content": f"Generate an image: {user_input}"}
                    ],
                    "modalities": model_cfg["modalities"],
                    "max_tokens": 4096,
                }
                
                logger.info(f"Image generation: trying {model_cfg['model']}...")
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                message = data.get("choices", [{}])[0].get("message", {})
                
                # Формат ответа: message.images[] содержит base64 data URL
                images = message.get("images", [])
                content = message.get("content", "")
                
                result_parts = []
                
                # Извлекаем изображения
                for img in images:
                    if isinstance(img, dict):
                        img_url = img.get("image_url", {}).get("url", "")
                        if img_url:
                            result_parts.append(f"![image]({img_url})")
                
                # Текстовый контент (если есть)
                if content:
                    result_parts.append(content)
                
                if result_parts:
                    logger.info(f"Image generation via {model_cfg['model']}: OK ({len(images)} images)")
                    return "\n\n".join(result_parts)
                
            except Exception as e:
                logger.warning(f"Image model {model_cfg['model']} failed: {e}")
                continue
        
        # Fallback: текстовое описание
        try:
            from grapharchitect.tools.ApiTools.OpenRouterTool.openrouter_llm import OpenRouterLLM
            
            llm = OpenRouterLLM(
                api_key=api_key,
                model_name="openai/gpt-3.5-turbo",
                system_prompt=(
                    "Ты художник. Подробно опиши изображение по запросу: "
                    "композицию, цвета, объекты, стиль. Отвечай на русском."
                )
            )
            description = llm.query_llm(
                question=f"Опиши изображение: {user_input}",
                temperature=0.7,
                max_tokens=500
            )
            return f"[Модель не поддерживает генерацию изображений. Текстовое описание:]\n\n{description}"
        except Exception as e:
            logger.error(f"Image fallback error: {e}")
            return f"[{self.metadata.tool_name}] Ошибка генерации: {user_input[:100]}"
    
    @property
    def agent_id(self) -> str:
        """ID оригинального агента из agent_library"""
        return self._agent_id
    
    @property
    def original_agent(self) -> Agent:
        """Оригинальный Agent объект"""
        return self._agent


class GraphArchitectBridge:
    """
    Главный класс интеграции GraphArchitect с Web API.
    
    Предоставляет:
    - Конверсию всех Agent → BaseTool
    - Реальный поиск стратегий в графе
    - Выполнение через ExecutionOrchestrator
    - Выбор инструментов через InstrumentSelector (softmax + температура)
    - Стриминг прогресса выполнения
    - Сбор данных для обучения
    """
    
    def __init__(self):
        logger.info("Initializing GraphArchitectBridge...")
        
        # Создание сервиса эмбеддингов через фабрику (поддержка Infinity)
        try:
            from grapharchitect.services.embedding.embedding_factory import create_embedding_service
            import config
            
            self.embedding_service = create_embedding_service(
                embedding_type=config.EMBEDDING_TYPE,
                dimension=config.EMBEDDING_DIMENSION,
                infinity_url=config.INFINITY_BASE_URL,
                infinity_api_key=config.INFINITY_API_KEY,
                infinity_model=config.INFINITY_MODEL,
                infinity_timeout=config.INFINITY_TIMEOUT,
                fallback_to_simple=True
            )
            logger.info(f"Embedding service created: {self.embedding_service.__class__.__name__}")
        
        except Exception as e:
            logger.error(f"Error creating embedding service from config: {e}")
            logger.info("Falling back to SimpleEmbeddingService")
            from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
            self.embedding_service = SimpleEmbeddingService(dimension=384)
        
        # Инициализация других сервисов
        self.selector = InstrumentSelector(temperature_constant=config.TEMPERATURE_CONSTANT)
        self.strategy_finder = GraphStrategyFinder()
        self.orchestrator = ExecutionOrchestrator(
            self.embedding_service,
            self.selector,
            self.strategy_finder
        )
        
        # NLI для парсинга задач
        # Если есть OpenRouter ключ — используем LLM NLI (точнее)
        # Иначе — k-NN ретривер (быстрее, но обобщённые коннекторы)
        self.nli = self._create_nli_service()
        self._load_nli_examples()
        
        # Обучение (опционально)
        self.training = TrainingOrchestrator(learning_rate=0.01)
        self.critic = SimpleCritic()
        
        # ReWOO Planning (использует OpenRouter API ключ)
        self.rewoo_planner = None
        if REWOO_AVAILABLE:
            try:
                import os
                api_key = os.getenv("OPENROUTER_API_KEY")
                if api_key:
                    self.rewoo_planner = ReWOOPlanner(
                        gemini_api_key=api_key  # Используем OpenRouter ключ
                    )
                    logger.info("ReWOO Planner initialized with OpenRouter API key")
                else:
                    logger.info("ReWOO Planner: waiting for API key (will use fallback decomposition)")
            except Exception as e:
                logger.warning(f"ReWOO Planner initialization error: {e}")
        
        # Конвертация всех агентов в инструменты
        self.tools = self._convert_agents_to_tools()
        self.agent_to_tool_map: Dict[str, AgentTool] = {}
        
        for tool in self.tools:
            if isinstance(tool, AgentTool):
                self.agent_to_tool_map[tool.agent_id] = tool
        
        logger.info(f"GraphArchitectBridge ready ({len(self.tools)} tools)")
    
    def _convert_agents_to_tools(self) -> List[BaseTool]:
        """Конвертировать всех агентов из БД в BaseTool"""
        repo = get_repository()
        agents = repo.get_all_agents()
        tools = []
        
        logger.info(f"Converting {len(agents)} tools to BaseTool...")
        
        for agent in agents:
            tool = AgentTool(agent)
            
            # Создаем эмбеддинг возможностей инструмента
            tool.metadata.capabilities_embedding = self.embedding_service.embed_tool_capabilities(tool)
            
            tools.append(tool)
            logger.debug(f"  Tool: {tool.metadata.tool_name} [{agent.type}] "
                        f"{tool.input.format} -> {tool.output.format}")
        
        # Сводка по типам
        type_counts = {}
        for t in tools:
            fmt = f"{t.input.format} -> {t.output.format}"
            type_counts[fmt] = type_counts.get(fmt, 0) + 1
        logger.info(f"Tool connector types: {type_counts}")
        
        return tools
    
    def _create_nli_service(self):
        """
        Создать NLI сервис.
        
        Приоритеты:
        1. LLM NLI через OpenRouter (если есть API ключ) — точный парсинг
        2. k-NN NLI с Faiss/наивным ретривером — быстрый, но обобщённый
        
        Returns:
            NLI сервис (LLMNLIService или NaturalLanguageInterface)
        """
        import os
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Попытка создать LLM NLI (если есть ключ OpenRouter)
        if api_key:
            try:
                from grapharchitect.services.nli.llm_nli_service import LLMNLIService
                
                llm_nli = LLMNLIService(
                    embedding_service=self.embedding_service,
                    backend="openrouter",
                    model_name="openai/gpt-3.5-turbo",
                    api_key=api_key,
                    k_similar=3,
                    temperature=0.1
                )
                
                if llm_nli.is_available():
                    logger.info("NLI: LLM через OpenRouter (точный парсинг коннекторов)")
                    return llm_nli
            
            except Exception as e:
                logger.warning(f"LLM NLI not available: {e}")
        
        # Fallback: k-NN NLI
        return self._create_knn_nli()
    
    def _create_knn_nli(self):
        """
        Создать k-NN NLI (fallback если LLM недоступен).
        
        Returns:
            NaturalLanguageInterface с k-NN ретривером
        """
        try:
            from grapharchitect.services.nli.retriever_factory import create_knn_retriever
            import config
            
            retriever = create_knn_retriever(
                embedding_service=self.embedding_service,
                retriever_type=config.KNN_TYPE,
                vector_weight=config.KNN_VECTOR_WEIGHT,
                text_weight=config.KNN_TEXT_WEIGHT,
                faiss_index_type=config.FAISS_INDEX_TYPE
            )
            
            from grapharchitect.services.nli.natural_language_interface import NaturalLanguageInterface
            nli = NaturalLanguageInterface(self.embedding_service, retriever=retriever)
            
            logger.info(f"NLI: k-NN с {retriever.__class__.__name__}")
            return nli
        
        except Exception as e:
            logger.error(f"Error creating k-NN NLI: {e}")
            logger.info("Falling back to default NLI")
            return NaturalLanguageInterface(self.embedding_service)
    
    def _load_nli_examples(self):
        """Загрузить примеры NLI из файла и вычислить эмбеддинги."""
        try:
            import json
            examples_file = Path(__file__).parent / "data" / "nli_examples.json"
            
            if examples_file.exists():
                with open(examples_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                examples = [NLIDatasetItem(**item) for item in data]
                
                # Вычисляем эмбеддинги для k-NN поиска (если не заданы)
                computed = 0
                for ex in examples:
                    if not ex.task_embedding and ex.task_text:
                        ex.task_embedding = self.embedding_service.embed_text(ex.task_text)
                        computed += 1
                
                self.nli.load_dataset(examples)
                logger.info(f"Loaded {len(examples)} NLI examples ({computed} embeddings computed)")
            else:
                logger.warning(f"NLI examples not found: {examples_file}")
        except Exception as e:
            logger.error(f"Error loading NLI examples: {e}")
    
    def _save_base64_images_to_files(self, text: str) -> str:
        """
        Найти base64-изображения в тексте и сохранить их в файлы.
        Заменяет data:image/...;base64,... на /uploads/имя_файла.png
        
        Это решает проблему передачи огромных base64-строк через JSON-стрим,
        где они разбиваются на части и ломают JSON.parse().
        """
        import re
        import base64
        
        pattern = r'!\[([^\]]*)\]\((data:image\/([a-zA-Z]+);base64,([^\s\)]+))\)'
        
        def replace_match(match):
            alt_text = match.group(1)
            mime_ext = match.group(3)  # png, jpeg, etc.
            b64_data = match.group(4)
            
            try:
                # Декодируем base64
                image_bytes = base64.b64decode(b64_data)
                
                # Генерируем имя файла
                filename = f"gen_{uuid.uuid4().hex[:12]}.{mime_ext}"
                
                # Сохраняем в uploads/
                upload_dir = Path(__file__).parent / "uploads"
                upload_dir.mkdir(exist_ok=True)
                filepath = upload_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                logger.info(f"Saved image: {filepath} ({len(image_bytes)} bytes)")
                return f"![{alt_text}](/uploads/{filename})"
            
            except Exception as e:
                logger.error(f"Failed to save image: {e}")
                return match.group(0)  # Возвращаем оригинал
        
        if "data:image" in text:
            result = re.sub(pattern, replace_match, text)
            return result
        
        return text
    
    def get_tool_by_agent_id(self, agent_id: str) -> Optional[AgentTool]:
        """Получить BaseTool по ID агента"""
        return self.agent_to_tool_map.get(agent_id)
    
    def get_tools_by_agent_ids(self, agent_ids: List[str]) -> List[BaseTool]:
        """Получить список BaseTool по списку ID агентов"""
        tools = []
        for agent_id in agent_ids:
            tool = self.get_tool_by_agent_id(agent_id)
            if tool:
                tools.append(tool)
        return tools
    
    async def parse_user_message(self, message: str) -> Tuple[Connector, Connector]:
        """
        Парсинг пользовательского сообщения через NLI.
        
        Преобразует текст задачи в пару коннекторов (входной, выходной).
        Поддерживает результаты от LLMNLIService (Connector) и k-NN NLI (ConnectorDescriptor).
        """
        try:
            result = self.nli.parse_task(message, self.tools, k=3)
            
            if result.success and result.task_representation:
                raw_input = result.task_representation.input_connector
                raw_output = result.task_representation.output_connector
                
                # Определяем тип объекта и конвертируем
                input_conn = self._to_connector(raw_input)
                output_conn = self._to_connector(raw_output)
                
                logger.info(f"NLI [{self.nli.__class__.__name__}]: "
                           f"'{message[:50]}' -> {input_conn.format} -> {output_conn.format}")
                return (input_conn, output_conn)
        
        except Exception as e:
            logger.error(f"NLI error: {e}", exc_info=True)
        
        # Fallback: default connectors
        logger.warning("NLI failed, using default connectors: text|question -> text|answer")
        return (
            Connector("text", "question"),
            Connector("text", "answer")
        )
    
    def _to_connector(self, obj) -> Connector:
        """
        Конвертировать объект в Connector.
        
        Поддерживает:
        - Connector (от LLMNLIService) — возвращает как есть
        - ConnectorDescriptor (от k-NN NLI) — извлекает data_type и semantic_type
        - None — fallback на text|data
        """
        if obj is None:
            return Connector("text", "data")
        
        # Уже Connector (от LLMNLIService)
        if isinstance(obj, Connector):
            return obj
        
        # ConnectorDescriptor (от k-NN NLI)
        data_format = "text"
        semantic_format = "data"
        
        if hasattr(obj, 'data_type') and obj.data_type:
            data_format = getattr(obj.data_type, 'subtype', None) or \
                          getattr(obj.data_type, 'complex_type', None) or "text"
        
        if hasattr(obj, 'semantic_type') and obj.semantic_type:
            semantic_format = getattr(obj.semantic_type, 'semantic_category', None) or "data"
        
        return Connector(data_format, semantic_format)
    
    async def find_strategies(
        self,
        start_format: str,
        end_format: str,
        algorithm: str = "yen_5"
    ) -> List[List[BaseTool]]:
        """
        Найти стратегии в графе (РЕАЛЬНЫЙ поиск через алгоритмы).
        
        Args:
            start_format: Входной формат (например "text|question")
            end_format: Выходной формат (например "text|answer")
            algorithm: Название алгоритма (yen_5, dijkstra и т.д.)
        
        Returns:
            Список стратегий (каждая стратегия = последовательность инструментов)
        """
        # Маппинг названий алгоритмов → PathfindingAlgorithm enum
        algo_map = {
            "dijkstra": PathfindingAlgorithm.DIJKSTRA,
            "astar": PathfindingAlgorithm.ASTAR,
            "yen_3": PathfindingAlgorithm.YEN,
            "yen_5": PathfindingAlgorithm.YEN,
            "yen_10": PathfindingAlgorithm.YEN,
            "ant_3": PathfindingAlgorithm.ANT_COLONY,
            "ant_5": PathfindingAlgorithm.ANT_COLONY,
            "ant_10": PathfindingAlgorithm.ANT_COLONY,
        }
        
        # Маппинг limit (количество путей для поиска)
        limit_map = {
            "yen_3": 3, "yen_5": 5, "yen_10": 10,
            "ant_3": 3, "ant_5": 5, "ant_10": 10,
            "dijkstra": 1, "astar": 1
        }
        
        algo = algo_map.get(algorithm, PathfindingAlgorithm.YEN)
        limit = limit_map.get(algorithm, 5)
        
        logger.info(f"Graph search: '{start_format}' -> '{end_format}' ({algorithm}, limit={limit})")
        
        # Диагностика: какие форматы есть в инструментах
        all_input_formats = set()
        all_output_formats = set()
        for t in self.tools:
            all_input_formats.add(t.input.format)
            all_output_formats.add(t.output.format)
        
        logger.info(f"Available input formats: {sorted(all_input_formats)}")
        logger.info(f"Available output formats: {sorted(all_output_formats)}")
        
        if start_format not in all_input_formats:
            logger.warning(f"Start format '{start_format}' NOT in tool inputs!")
        if end_format not in all_output_formats:
            logger.warning(f"End format '{end_format}' NOT in tool outputs!")
        
        # Реальный поиск в графе
        strategies = self.strategy_finder.find_strategies(
            self.tools,
            start_format,
            end_format,
            limit=limit,
            algorithm=algo
        )
        
        logger.info(f"Found {len(strategies)} strategies for {start_format} -> {end_format}")
        
        return strategies
    
    async def select_tool_from_group(
        self,
        tool_group: List[BaseTool],
        task_embedding: Optional[List[float]] = None,
        top_k: int = 5
    ):
        """
        Выбрать инструмент из группы через softmax с температурой.
        
        Это РЕАЛЬНЫЙ алгоритм выбора, заменяющий random в WorkflowSimulator.
        
        Returns:
            InstrumentSelectionResult с selected_tool, probabilities, temperature
        """
        if not tool_group:
            return None
        
        selection_result = self.selector.select_instrument(
            tool_group,
            task_embedding,
            top_k=min(top_k, len(tool_group))
        )
        
        logger.debug(f"Selected: {selection_result.selected_tool.metadata.tool_name} "
              f"(p={selection_result.selection_probability:.3f}, T={selection_result.temperature:.3f})")
        
        return selection_result
    
    async def execute_task_full(
        self,
        message: str,
        input_data: any,
        algorithm: str = "yen_5",
        top_k: int = 5
    ) -> ExecutionContext:
        """
        Полное выполнение задачи через ExecutionOrchestrator.
        
        Возвращает ExecutionContext со всеми метриками и градиентными трассами.
        """
        # 1. Парсинг задачи (NLI или дефолт)
        input_conn, output_conn = await self.parse_user_message(message)
        
        # 2. Создание TaskDefinition
        task = TaskDefinition(
            description=message,
            input_connector=input_conn,
            output_connector=output_conn,
            input_data=input_data
        )
        
        # 3. Создание эмбеддинга задачи
        task.task_embedding = self.embedding_service.embed_text(message)
        
        # 4. Выполнение через оркестратор
        logger.info(f"Executing task: {message[:50]}...")
        
        context = self.orchestrator.execute_task(
            task,
            self.tools,
            path_limit=5,
            top_k=top_k
        )
        
        logger.info(f"Task completed: {context.status.value}")
        logger.info(f"Steps: {context.get_total_steps()}, Time: {context.total_time:.2f}s, Cost: {context.total_cost:.2f}")
        
        # 5. Автоматическая оценка и обучение
        await self._auto_evaluate_and_train(context)
        
        return context
    
    async def execute_task_streaming(
        self,
        message: str,
        input_data: any,
        algorithm: str = "yen_5",
        top_k: int = 5,
        use_rewoo: bool = False,
        user_priority: str = "balanced",
        max_cost: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> AsyncGenerator[MessageChunk, None]:
        """
        Выполнить задачу с real-time стримингом прогресса.
        
        Yields MessageChunk для каждого этапа выполнения.
        Клиент получает обновления о:
        - Поиске стратегий
        - Выборе инструментов
        - Выполнении каждого шага
        - Финальном результате
        """
        # 0. Адаптация под профиль пользователя
        priority_labels = {
            "speed": "скорость",
            "quality": "качество",
            "cost": "экономия",
            "balanced": "баланс"
        }
        
        yield MessageChunk(
            type="gen_phase_start",
            phase_id="user_adaptation",
            content=f"Адаптация: приоритет = {priority_labels.get(user_priority, user_priority)}"
        )
        
        await asyncio.sleep(0.1)
        
        # Применяем приоритет пользователя к температуре
        temp_multipliers = {
            "speed": 0.5,
            "quality": 0.3,
            "cost": 0.7,
            "balanced": 1.0
        }
        
        effective_temp = temp_multipliers.get(user_priority, 1.0)
        
        yield MessageChunk(
            type="gen_phase_complete",
            phase_id="user_adaptation",
            metadata={
                "priority": user_priority,
                "temperature_multiplier": effective_temp,
                "max_cost": max_cost,
                "max_time": max_time
            }
        )
        
        # 1. Парсинг задачи через NLI
        yield MessageChunk(
            type="gen_phase_start",
            phase_id="nli_parsing",
            content="Анализ задачи через NLI..."
        )
        
        input_conn, output_conn = await self.parse_user_message(message)
        
        nli_type = self.nli.__class__.__name__
        connector_str = f"{input_conn.format} -> {output_conn.format}"
        
        yield MessageChunk(
            type="gen_phase_complete",
            phase_id="nli_parsing",
            content=f"NLI ({nli_type}): {connector_str}",
            metadata={
                "input_format": input_conn.format,
                "output_format": output_conn.format,
                "nli_type": nli_type,
                "connector_chain": connector_str
            }
        )
        
        await asyncio.sleep(0.2)
        
        # 2. Поиск стратегий в графе
        yield MessageChunk(
            type="gen_phase_start",
            phase_id="graph_search",
            content=f"Поиск путей в графе ({algorithm})..."
        )
        
        strategies = await self.find_strategies(
            input_conn.format,
            output_conn.format,
            algorithm
        )
        
        if not strategies:
            yield MessageChunk(
                type="error",
                content="No strategies found. Check tool connectors."
            )
            return
        
        yield MessageChunk(
            type="gen_phase_complete",
            phase_id="graph_search",
            metadata={
                "strategies_found": len(strategies),
                "strategy_length": len(strategies[0]) if strategies else 0
            }
        )
        
        await asyncio.sleep(0.3)
        
        # Вспомогательная функция: получить agent_id инструмента
        def _get_agent_id(tool) -> str:
            if isinstance(tool, AgentTool):
                return tool.agent_id
            return tool.metadata.tool_name
        
        # Вспомогательная функция: получить имя инструмента
        def _get_tool_name(tool) -> str:
            return tool.metadata.tool_name
        
        # 2.5 Формируем начальную структуру workflow (для стандартного пути)
        strategy = strategies[0]
        default_steps = []
        for idx, tool_or_edge in enumerate(strategy):
            step_name = f"Шаг {idx + 1}"
            if hasattr(tool_or_edge, 'tools'):
                candidates = [_get_agent_id(t) for t in tool_or_edge.tools]
            else:
                # Находим всех конкурентов
                primary = tool_or_edge
                in_fmt = primary.input.format
                out_fmt = primary.output.format
                all_matching = [primary]
                for t in self.tools:
                    if t is primary:
                        continue
                    if t.input.format == in_fmt and t.output.format == out_fmt:
                        all_matching.append(t)
                if len(all_matching) > 8:
                    all_matching.sort(key=lambda t: t.metadata.reputation, reverse=True)
                    all_matching = all_matching[:8]
                candidates = [_get_agent_id(t) for t in all_matching]
            default_steps.append({
                "id": f"step-{idx}",
                "name": step_name,
                "candidates": candidates
            })
        
        # Отправляем workflow_info чтобы фронтенд инициализировал currentWorkflow
        yield MessageChunk(
            type="workflow_info",
            metadata={
                "name": f"GraphArchitect ({algorithm})",
                "workflowId": f"ga_{id(self)}",
                "steps": default_steps
            }
        )
        
        await asyncio.sleep(0.2)
        
        # 3. ReWOO Planning (если включен)
        rewoo_plan = None
        if use_rewoo and self.rewoo_planner:
            yield MessageChunk(
                type="gen_phase_start",
                phase_id="rewoo_planning",
                content="ReWOO: Создание плана через LLM..."
            )
            
            # Сортируем инструменты по репутации (топ-10)
            sorted_tools = sorted(
                self.tools,
                key=lambda t: t.metadata.reputation,
                reverse=True
            )
            
            rewoo_plan = self.rewoo_planner.create_plan(
                task_description=message,
                strategies=strategies,
                algorithm_used=algorithm,
                top_tools=sorted_tools[:10]
            )
            
            if rewoo_plan:
                # Показываем план
                steps_info = []
                for step in rewoo_plan.steps:
                    steps_info.append(f"{step.tool_name}: {step.description}")
                
                yield MessageChunk(
                    type="gen_phase_complete",
                    phase_id="rewoo_planning",
                    metadata={
                        "steps_in_plan": len(rewoo_plan.steps),
                        "reasoning": rewoo_plan.reasoning[:300],
                        "steps": steps_info,
                        "estimated_time": rewoo_plan.estimated_time,
                        "estimated_cost": rewoo_plan.estimated_cost
                    }
                )
                
                # Обновляем workflow_info с шагами из ReWOO плана
                rewoo_steps = []
                for step_idx, plan_step in enumerate(rewoo_plan.steps):
                    # Находим кандидатов: по имени + все с такими же коннекторами
                    name_matched = []
                    for tool in self.tools:
                        tn = tool.metadata.tool_name
                        if (plan_step.tool_name.lower() in tn.lower() 
                            or tn.lower() in plan_step.tool_name.lower()):
                            name_matched.append(tool)
                    
                    all_candidates = list(name_matched)
                    if name_matched:
                        ref = name_matched[0]
                        for tool in self.tools:
                            if tool not in all_candidates:
                                if tool.input.format == ref.input.format and tool.output.format == ref.output.format:
                                    all_candidates.append(tool)
                    
                    if not all_candidates:
                        sorted_by_rep = sorted(self.tools, key=lambda t: t.metadata.reputation, reverse=True)
                        all_candidates = sorted_by_rep[:5]
                    
                    if len(all_candidates) > 6:
                        all_candidates.sort(key=lambda t: t.metadata.reputation, reverse=True)
                        all_candidates = all_candidates[:6]
                    
                    rewoo_steps.append({
                        "id": f"rewoo-step-{step_idx}",
                        "name": plan_step.description,
                        "candidates": [_get_agent_id(t) for t in all_candidates]
                    })
                
                # Переотправляем workflow_info с шагами ReWOO
                yield MessageChunk(
                    type="workflow_info",
                    metadata={
                        "name": f"ReWOO Plan ({algorithm})",
                        "workflowId": f"rewoo_{id(self)}",
                        "steps": rewoo_steps
                    }
                )
                
                await asyncio.sleep(0.3)
                
                # Выполняем по плану ReWOO с поддержкой зависимостей
                step_results = {}  # step_id → результат
                original_input = input_data
                current_data = input_data
                
                for step_idx, plan_step in enumerate(rewoo_plan.steps):
                    step_id = f"rewoo-step-{step_idx}"
                    is_last_step = (step_idx == len(rewoo_plan.steps) - 1)
                    
                    # Формируем вход на основе зависимостей
                    if plan_step.depends_on and len(plan_step.depends_on) > 0:
                        # Есть зависимости — собираем результаты зависимых шагов
                        dep_results = []
                        for dep_id in plan_step.depends_on:
                            # depends_on может быть "step-1" или "step-0" (нумерация из плана)
                            # Ищем по разным форматам ID
                            dep_result = step_results.get(dep_id)
                            if not dep_result:
                                # Пробуем rewoo-step-N формат
                                for key, val in step_results.items():
                                    if dep_id in key or key in dep_id:
                                        dep_result = val
                                        break
                            if dep_result:
                                dep_results.append(str(dep_result))
                        
                        if dep_results:
                            # Для финального объединения используем специальный промпт
                            if is_last_step:
                                current_data = f"{plan_step.description}\n\nВОТ ДАННЫЕ ДЛЯ ОБЪЕДИНЕНИЯ:\n" + "\n---\n".join(dep_results)
                            else:
                                current_data = f"{plan_step.description}\n\nКонтекст:\n" + "\n---\n".join(dep_results)
                        else:
                            current_data = plan_step.description
                    else:
                        # Нет зависимостей — используем описание шага как инструкцию
                        current_data = plan_step.description
                    
                    # Логируем что пойдет в инструмент
                    logger.info(f"Step {step_idx+1} final instruction: {current_data[:200]}...")
                    
                    # Находим кандидатов:
                    # 1) По имени из плана
                    # 2) Добавляем ВСЕ инструменты с совместимыми коннекторами
                    name_matched = []
                    for tool in self.tools:
                        tn = tool.metadata.tool_name
                        if (plan_step.tool_name.lower() in tn.lower()
                            or tn.lower() in plan_step.tool_name.lower()):
                            name_matched.append(tool)
                    
                    # Расширяем: добавляем инструменты с такими же коннекторами
                    candidates = list(name_matched)
                    if name_matched:
                        ref_tool = name_matched[0]
                        ref_in = ref_tool.input.format
                        ref_out = ref_tool.output.format
                        for tool in self.tools:
                            if tool not in candidates:
                                if tool.input.format == ref_in and tool.output.format == ref_out:
                                    candidates.append(tool)
                    
                    if not candidates:
                        # Fallback: берем топ-5 по репутации
                        sorted_by_rep = sorted(
                            self.tools,
                            key=lambda t: t.metadata.reputation,
                            reverse=True
                        )
                        candidates = sorted_by_rep[:5]
                    
                    # Ограничиваем до 6 для читаемости
                    if len(candidates) > 6:
                        candidates.sort(key=lambda t: t.metadata.reputation, reverse=True)
                        candidates = candidates[:6]
                    
                    # Событие начала шага
                    yield MessageChunk(
                        type="step_started",
                        step_id=step_id,
                        metadata={
                            "name": plan_step.description,
                            "candidates": [_get_agent_id(t) for t in candidates]
                        }
                    )
                    
                    await asyncio.sleep(0.3)
                    
                    # СОРЕВНОВАНИЕ: минимум 2 секунды анимации
                    # Показываем даже для 1 кандидата (визуализация работы)
                    competition_steps = 10  # 10 шагов по 0.2с = 2 секунды
                    for progress_step in range(competition_steps + 1):
                        progress = int(progress_step * 100 / competition_steps)
                        await asyncio.sleep(0.2)
                        
                        for tool in candidates:
                            yield MessageChunk(
                                type="agent_progress",
                                agent_id=_get_agent_id(tool),
                                step_id=step_id,
                                progress=progress
                            )
                        
                        # Обновляем scores
                        if progress >= 20 and len(candidates) > 1:
                            scores = {}
                            for tool in candidates:
                                base_score = tool.metadata.reputation
                                noise = (hash(tool.metadata.tool_name + str(progress)) % 100) / 1000
                                current_score = base_score * (0.8 + progress / 400) + noise
                                scores[tool] = min(current_score, 0.99)
                            
                            yield MessageChunk(
                                type="agent_score_updated",
                                step_id=step_id,
                                metadata={
                                    "agents": [
                                        {
                                            "agentId": _get_agent_id(t),
                                            "score": round(scores[t], 3)
                                        }
                                        for t in candidates
                                    ]
                                }
                            )
                    
                    # Выбор через softmax
                    if len(candidates) > 1:
                        selection = await self.select_tool_from_group(
                            candidates,
                            self.embedding_service.embed_text(message),
                            top_k=min(5, len(candidates))
                        )
                        matched_tool = selection.selected_tool if selection else candidates[0]
                        prob = selection.selection_probability if selection else 1.0
                    else:
                        matched_tool = candidates[0]
                        prob = 1.0
                    
                    winner_id = _get_agent_id(matched_tool)
                    
                    yield MessageChunk(
                        type="agent_selected",
                        agent_id=winner_id,
                        step_id=step_id,
                        score=prob
                    )
                    
                    await asyncio.sleep(0.5)
                    
                    # Выполнение победителя
                    yield MessageChunk(
                        type="agent_executing",
                        agent_id=winner_id,
                        step_id=step_id,
                        progress=30,
                        content=plan_step.description
                    )
                    
                    await asyncio.sleep(0.3)
                    
                    # Лог входных данных
                    input_preview = str(current_data)[:150].replace('\n', ' ')
                    logger.info(f"ReWOO step {step_idx+1} INPUT [{matched_tool.metadata.tool_name}]: {input_preview}")
                    
                    current_data = matched_tool.execute(current_data)
                    
                    # Лог выходных данных
                    output_preview = str(current_data)[:150].replace('\n', ' ')
                    logger.info(f"ReWOO step {step_idx+1} OUTPUT [{matched_tool.metadata.tool_name}]: {output_preview}")
                    
                    # Сохраняем результат шага для зависимостей
                    step_results[step_id] = current_data
                    step_results[plan_step.step_id] = current_data  # Дублируем под ID из плана
                    
                    yield MessageChunk(
                        type="agent_executing",
                        agent_id=winner_id,
                        step_id=step_id,
                        progress=100,
                        content="Завершено"
                    )
                    
                    # Промежуточный результат в UI лог
                    result_str = str(current_data)
                    if "data:image" in result_str or "![image]" in result_str:
                        result_preview = "[IMAGE GENERATED]"
                    else:
                        result_preview = result_str[:200].replace('\n', ' ')
                    
                    yield MessageChunk(
                        type="gen_phase_start",
                        phase_id=f"rewoo_result_{step_idx}",
                        content=f"[Результат шага {step_idx+1}] {result_preview}"
                    )
                    
                    yield MessageChunk(
                        type="step_completed",
                        step_id=step_id,
                        metadata={"result_preview": str(current_data)[:200]}
                    )
                    
                    await asyncio.sleep(0.3)
                
                # Финальный результат — вывод последнего шага (он уже объединённый)
                final_result = str(current_data)
                
                # Если результат содержит base64-изображение, сохраняем в файл
                final_result = self._save_base64_images_to_files(final_result)
                
                logger.info(f"ReWOO Final result: {final_result[:200]}...")
                
                yield MessageChunk(type="text", content=final_result)
                return  # Завершаем - ReWOO план выполнен
            
            else:
                yield MessageChunk(
                    type="gen_phase_complete",
                    phase_id="rewoo_planning",
                    content="ReWOO plan не создан, используется стандартный путь"
                )
            
            await asyncio.sleep(0.3)
        
        # Берем первую (лучшую) стратегию
        strategy = strategies[0]
        
        yield MessageChunk(
            type="gen_phase_start",
            phase_id="strategy_selected",
            content=f"Выбрана стратегия из {len(strategy)} шагов" + 
                   (f" (ReWOO план: {len(rewoo_plan.steps)} шагов)" if rewoo_plan else "")
        )
        
        await asyncio.sleep(0.2)
        
        # 4. Создаем задачу
        task = TaskDefinition(
            description=message,
            input_connector=input_conn,
            output_connector=output_conn,
            input_data=input_data
        )
        
        task.task_embedding = self.embedding_service.embed_text(message)
        
        # 5. Подготовка данных для цепочки
        # Если конечный формат — изображение, а первый шаг — текстовый,
        # модифицируем вход чтобы QA/Writer создал ОПИСАНИЕ для генератора
        current_data = input_data
        if (output_conn.data_format == "image" and 
            len(strategy) > 1 and 
            input_conn.data_format == "text"):
            current_data = (
                f"Создай подробное текстовое описание изображения по запросу пользователя. "
                f"Опиши: объекты, композицию, цвета, стиль, фон. "
                f"Запрос: {input_data}"
            )
            logger.info(f"Image pipeline: modified input for text→image chain")
        
        gradient_traces = []
        
        for step_index, tool_or_edge in enumerate(strategy):
            step_id = f"step-{step_index}"
            
            # Определяем группу инструментов
            if hasattr(tool_or_edge, 'tools'):
                # Это ToolEdge с группой инструментов
                tool_group = tool_or_edge.tools
            else:
                # Один инструмент — но ищем ВСЕХ конкурентов с такими же коннекторами
                primary_tool = tool_or_edge
                in_fmt = primary_tool.input.format
                out_fmt = primary_tool.output.format
                
                # Находим все инструменты с совпадающими коннекторами
                tool_group = [primary_tool]
                for t in self.tools:
                    if t is primary_tool:
                        continue
                    if t.input.format == in_fmt and t.output.format == out_fmt:
                        tool_group.append(t)
                
                # Ограничиваем до топ-8 по репутации (чтобы UI не перегрузить)
                if len(tool_group) > 8:
                    tool_group.sort(key=lambda t: t.metadata.reputation, reverse=True)
                    tool_group = tool_group[:8]
            
            # Событие начала шага (с agent_id для корректной работы фронтенда)
            yield MessageChunk(
                type="step_started",
                step_id=step_id,
                content=f"Шаг {step_index + 1}/{len(strategy)}",
                metadata={
                    "name": f"Шаг {step_index + 1}",
                    "candidates": [_get_agent_id(t) for t in tool_group],
                    "candidates_count": len(tool_group)
                }
            )
            
            await asyncio.sleep(0.2)
            
            # Визуализация соревнования: минимум 2 секунды
            competition_steps = 10  # 10 * 0.2с = 2 секунды минимум
            for progress_step in range(competition_steps + 1):
                progress = int(progress_step * 100 / competition_steps)
                await asyncio.sleep(0.2)
                
                for tool in tool_group:
                    yield MessageChunk(
                        type="agent_progress",
                        agent_id=_get_agent_id(tool),
                        step_id=step_id,
                        progress=progress
                    )
                
                # Обновляем scores
                if progress >= 20 and len(tool_group) > 1:
                    scores = {}
                    for tool in tool_group:
                        base_score = tool.metadata.reputation
                        noise = (hash(tool.metadata.tool_name + str(progress)) % 100) / 1000
                        current_score = base_score * (0.8 + progress / 400) + noise
                        scores[tool] = min(current_score, 0.99)
                    
                    yield MessageChunk(
                        type="agent_score_updated",
                        step_id=step_id,
                        metadata={
                            "agents": [
                                {
                                    "agentId": _get_agent_id(t),
                                    "score": round(scores[t], 3)
                                }
                                for t in tool_group
                            ]
                        }
                    )
            
            # Выбор инструмента через РЕАЛЬНЫЙ softmax!
            selection_result = await self.select_tool_from_group(
                tool_group,
                task.task_embedding,
                top_k=min(5, len(tool_group))
            )
            
            if not selection_result:
                yield MessageChunk(type="error", content="Ошибка выбора инструмента")
                return
            
            selected_tool = selection_result.selected_tool
            
            # Финальные scores с реальными вероятностями от softmax
            yield MessageChunk(
                type="agent_score_updated",
                step_id=step_id,
                metadata={
                    "agents": [
                        {
                            "agentId": t.agent_id if isinstance(t, AgentTool) else t.metadata.tool_name,
                            "score": round(selection_result.all_probabilities.get(t, 0), 3),
                            "logit": round(selection_result.all_logits.get(t, 0), 3)
                        }
                        for t in tool_group
                    ],
                    "temperature": round(selection_result.temperature, 3)
                }
            )
            
            # Агент выбран
            agent_id = selected_tool.agent_id if isinstance(selected_tool, AgentTool) else selected_tool.metadata.tool_name
            
            yield MessageChunk(
                type="agent_selected",
                agent_id=agent_id,
                step_id=step_id,
                score=selection_result.selection_probability,
                metadata={
                    "temperature": selection_result.temperature,
                    "top_k": selection_result.top_k
                }
            )
            
            await asyncio.sleep(0.3)
            
            # Выполнение инструмента
            yield MessageChunk(
                type="agent_executing",
                agent_id=agent_id,
                step_id=step_id,
                progress=30,
                content="Обработка данных..."
            )
            
            await asyncio.sleep(0.2)
            
            try:
                # Лог: что получает инструмент на вход
                input_preview = str(current_data)[:150].replace('\n', ' ')
                logger.info(f"Step {step_index+1} INPUT [{selected_tool.metadata.tool_name}]: {input_preview}")
                
                # РЕАЛЬНОЕ выполнение!
                current_data = selected_tool.execute(current_data)
                
                # Лог: что инструмент выдал
                output_preview = str(current_data)[:150].replace('\n', ' ')
                logger.info(f"Step {step_index+1} OUTPUT [{selected_tool.metadata.tool_name}]: {output_preview}")
                
                yield MessageChunk(
                    type="agent_executing",
                    agent_id=agent_id,
                    step_id=step_id,
                    progress=100,
                    content="Завершено"
                )
            
            except Exception as e:
                yield MessageChunk(
                    type="error",
                    content=f"Ошибка выполнения: {str(e)}"
                )
                return
            
            await asyncio.sleep(0.2)
            
            # Промежуточный результат в лог
            result_str = str(current_data) if current_data else "(пусто)"
            if "data:image" in result_str or "![image]" in result_str:
                result_preview = "[IMAGE GENERATED]"
            else:
                result_preview = result_str[:200].replace('\n', ' ')
            
            yield MessageChunk(
                type="gen_phase_start",
                phase_id=f"step_result_{step_index}",
                content=f"[Результат шага {step_index+1}] {result_preview}"
            )
            
            # Шаг завершен
            yield MessageChunk(
                type="step_completed",
                step_id=step_id,
                metadata={
                    "result_preview": str(current_data)[:200] if current_data else None
                }
            )
            
            # Сохраняем градиентную трассу
            gradient_traces.append(selection_result.gradient_info)
            
            await asyncio.sleep(0.3)
        
        # 6. RLAIF оценка (если есть OpenRouter ключ) - ОПЦИОНАЛЬНО, не блокирует ответ
        import os
        rlaif_score = None
        
        if os.getenv("OPENROUTER_API_KEY") and len(str(current_data)) < 50000:
            try:
                from grapharchitect.services.rlaif.llm_critic import LLMCritic
                
                yield MessageChunk(
                    type="gen_phase_start",
                    phase_id="rlaif_evaluation",
                    content="RLAIF: Оценка качества..."
                )
                
                # Очищаем ответ от тяжелых бинарных данных перед оценкой
                eval_answer = str(current_data)
                if "data:image" in eval_answer or len(eval_answer) > 5000:
                    import re
                    eval_answer = re.sub(r'data:image\/[a-zA-Z]*;base64,[^\s\)]*', '[IMAGE]', eval_answer)
                    if len(eval_answer) > 2000:
                        eval_answer = eval_answer[:2000] + "..."
                
                critic_llm = LLMCritic(
                    backend="openrouter",
                    model_name="openai/gpt-3.5-turbo",
                    temperature=0.2,
                    detailed_evaluation=False  # Упрощенная оценка
                )
                
                rlaif_result = critic_llm.evaluate_answer(
                    task=message,
                    answer=eval_answer,
                    context={"tools_used": []}
                )
                
                rlaif_score = rlaif_result.overall_score
                
                yield MessageChunk(
                    type="gen_phase_complete",
                    phase_id="rlaif_evaluation",
                    metadata={"overall_score": round(rlaif_score, 2)}
                )
                
                logger.info(f"RLAIF score: {rlaif_score:.2f}")
            
            except Exception as e:
                logger.warning(f"RLAIF evaluation skipped: {e}")
                yield MessageChunk(
                    type="gen_phase_complete",
                    phase_id="rlaif_evaluation",
                    content="Пропущено"
                )
        
        # 7. Финальный результат (без "Execution result:" — чистый ответ)
        result_text = str(current_data)
        
        # Если результат содержит base64-изображение, сохраняем в файл
        result_text = self._save_base64_images_to_files(result_text)
        
        # Логируем финальный результат в консоль
        if "/static/" in result_text or "/uploads/" in result_text:
            logger.info(f"Final result: [IMAGE saved to file]")
        else:
            logger.info(f"Final result: {result_text[:200]}...")
        
        yield MessageChunk(
            type="text",
            content=result_text
        )
    
    async def _auto_evaluate_and_train(self, context: ExecutionContext):
        """
        Автоматическая оценка и обучение после выполнения.
        
        Вызывается автоматически после каждого выполнения.
        Сохраняет данные в SQLite БД (если доступна).
        """
        try:
            # Автоматическая оценка через SimpleCritic
            feedback = self.critic.evaluate_execution(context)
            
            logger.info(f"Auto-evaluation: {feedback.quality_score:.2f}")
            
            # Добавление в датасет для обучения
            self.training.add_execution_to_dataset(context, [feedback])
            
            # Дообучение инструментов
            tools_to_train = [
                step.selected_tool
                for step in context.execution_steps
                if step.selected_tool
            ]
            
            if tools_to_train:
                self.training.train_all_tools(tools_to_train)
                logger.info(f"Trained tools: {len(tools_to_train)}")
                
                # Сохраняем обновленные метрики в БД
                await self._save_tool_metrics_to_db(tools_to_train)
            
            # Сохраняем историю выполнения в БД
            await self._save_execution_to_db(context, feedback)
        
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    async def _save_tool_metrics_to_db(self, tools: List[BaseTool]):
        """Сохранить метрики инструментов в БД"""
        try:
            from sqlite_repository import get_sqlite_repository
            repo = get_sqlite_repository()
            
            for tool in tools:
                if isinstance(tool, AgentTool):
                    repo.save_tool_metrics(
                        agent_id=tool.agent_id,
                        tool_name=tool.metadata.tool_name,
                        reputation=tool.metadata.reputation,
                        mean_cost=tool.metadata.mean_cost,
                        mean_time=tool.metadata.mean_time_answer,
                        training_sample_size=tool.metadata.training_sample_size,
                        variance_estimate=tool.metadata.variance_estimate,
                        quality_scores=tool.metadata.quality_scores,
                        capabilities_embedding=tool.metadata.capabilities_embedding
                    )
            
            logger.debug("Metrics saved to database")
        
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    async def _save_execution_to_db(self, context: ExecutionContext, feedback):
        """Сохранить историю выполнения в БД"""
        try:
            from sqlite_repository import get_sqlite_repository
            repo = get_sqlite_repository()
            
            # Извлекаем данные из контекста
            selected_tools = [
                step.selected_tool.metadata.tool_name
                for step in context.execution_steps
                if step.selected_tool
            ]
            
            # Упрощенные градиентные трассы (только основное)
            gradient_traces = [
                {
                    'temperature': trace.temperature,
                    'selected_tool': trace.selected_tool.metadata.tool_name if trace.selected_tool else None,
                    'probabilities_count': len(trace.probabilities) if trace.probabilities else 0
                }
                for trace in context.gradient_traces
            ]
            
            input_format = context.task.input_connector.format if context.task else "unknown"
            output_format = context.task.output_connector.format if context.task else "unknown"
            
            repo.save_execution(
                execution_id=str(uuid.uuid4()),
                task_id=str(context.task_id),
                chat_id=None,  # TODO: передавать chat_id из контекста
                task_description=context.task.description if context.task else "",
                input_format=input_format,
                output_format=output_format,
                algorithm_used="auto",  # TODO: сохранять использованный алгоритм
                status=context.status.value,
                selected_tools=selected_tools,
                gradient_traces=gradient_traces,
                result=context.result,
                total_time=context.total_time,
                total_cost=context.total_cost
            )
            
            # Сохраняем feedback
            repo.save_feedback(
                task_id=str(context.task_id),
                execution_id=None,  # TODO: связать с execution
                source=feedback.source.value,
                quality_score=feedback.quality_score,
                success=feedback.success,
                comment=feedback.comment
            )
            
            logger.debug("Execution history saved to database")
        
        except Exception as e:
            logger.error(f"Failed to save execution history: {e}")
    
    async def submit_user_feedback(
        self,
        context: ExecutionContext,
        quality_score: float,
        comment: str = ""
    ):
        """
        Обработать пользовательскую обратную связь.
        
        Args:
            context: Контекст выполнения
            quality_score: Оценка качества (0.0-1.0)
            comment: Комментарий пользователя
        """
        feedback = FeedbackData(
            task_id=context.task_id,
            source=FeedbackSource.USER,
            quality_score=quality_score,
            comment=comment,
            success=quality_score >= 0.7
        )
        
        # Добавляем в датасет
        self.training.add_execution_to_dataset(context, [feedback])
        
        # Обучаем инструменты
        tools_to_train = [
            step.selected_tool
            for step in context.execution_steps
            if step.selected_tool
        ]
        
        if tools_to_train:
            self.training.train_all_tools(tools_to_train)
            logger.info(f"Training from user feedback: {quality_score:.2f}")
    
    def get_training_statistics(self):
        """Получить статистику обучения"""
        return self.training.get_statistics()


# ==================== Singleton instance ====================

_bridge: Optional[GraphArchitectBridge] = None
_bridge_error: Optional[Exception] = None


def get_bridge() -> GraphArchitectBridge:
    """
    Получить экземпляр GraphArchitectBridge (singleton).
    
    При первом вызове создает и инициализирует мост.
    При последующих - возвращает существующий экземпляр.
    """
    global _bridge, _bridge_error
    
    if _bridge is None and _bridge_error is None:
        try:
            print("\n" + "="*70)
            logger.info("Initializing GraphArchitect Bridge")
            print("="*70)
            
            _bridge = GraphArchitectBridge()
            
            print("="*70)
            logger.info("Bridge ready to use!")
            print("="*70 + "\n")
        
        except Exception as e:
            _bridge_error = e
            logger.error(f"Bridge initialization error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    if _bridge_error:
        raise RuntimeError(f"Bridge не инициализирован: {_bridge_error}")
    
    return _bridge


def is_bridge_available() -> bool:
    """Проверить доступность GraphArchitect Bridge"""
    try:
        get_bridge()
        return True
    except:
        return False
