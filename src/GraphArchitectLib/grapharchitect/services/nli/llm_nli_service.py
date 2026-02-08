"""
LLM-based NLI сервис.

Использует LLM (OpenRouter, VLLM, DeepSeek) для парсинга естественного языка
в структурированные коннекторы (dataType, semanticType, knowledgeDomain).

Основан на промпте из файнтюненной модели Ponimash/Qwen2.5-nli-7b.

Алгоритм:
1. Поиск похожих примеров через k-NN
2. Формирование few-shot промпта с описанием схемы коннекторов
3. Вызов LLM для получения JSON-структуры
4. Парсинг JSON и конвертация в Connector
"""

import logging
from typing import List, Optional, Dict, Any
import json
import os

from ...entities.base_tool import BaseTool
from ...entities.connectors.task_representation import TaskRepresentation
from ...entities.connectors.connector import Connector
from ..embedding.embedding_service import EmbeddingService
from .nli_dataset_item import NLIDatasetItem
from .natural_language_interface import NLIParseResult

logger = logging.getLogger(__name__)


class LLMNLIService:
    """
    NLI сервис на основе LLM.
    
    Использует LLM для точного парсинга с few-shot примерами из k-NN.
    Промпт повторяет логику файнтюненной Qwen NLI модели:
    задача -> JSON с input_connector и output_connector.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        backend: str = "openrouter",
        model_name: str = "openai/gpt-3.5-turbo",
        api_key: Optional[str] = None,
        vllm_host: Optional[str] = None,
        k_similar: int = 3,
        temperature: float = 0.1
    ):
        """
        Инициализация LLM NLI.
        
        Args:
            embedding_service: Сервис эмбеддингов для k-NN
            backend: Бэкенд LLM ("openrouter", "vllm", "deepseek")
            model_name: Название модели
            api_key: API ключ (для OpenRouter/DeepSeek)
            vllm_host: URL VLLM сервера
            k_similar: Количество похожих примеров для few-shot
            temperature: Температура генерации (низкая для точности)
        """
        self._embedding_service = embedding_service
        self._backend = backend
        self._model_name = model_name
        self._k_similar = k_similar
        self._temperature = temperature
        
        # Датасет примеров
        self._dataset: List[NLIDatasetItem] = []
        
        # Инициализация LLM бэкенда
        self._llm = None
        self._init_backend(api_key, vllm_host)
        
        logger.info(f"LLM NLI initialized: {backend} with {model_name}")
    
    def _init_backend(self, api_key: Optional[str], vllm_host: Optional[str]):
        """Инициализация LLM бэкенда."""
        
        if self._backend == "openrouter":
            from ...tools.ApiTools.OpenRouterTool.openrouter_llm import OpenRouterLLM
            
            self._llm = OpenRouterLLM(
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                model_name=self._model_name,
                system_prompt=(
                    "Ты эксперт по парсингу задач в структурированные коннекторы. "
                    "Всегда отвечай только JSON без пояснений."
                )
            )
            
            logger.info("OpenRouter backend initialized for NLI")
        
        elif self._backend == "vllm":
            from ...tools.ApiTools.VLLMTool.VLLMApi import VLLMApi
            
            vllm_host = vllm_host or os.getenv("VLLM_HOST", "http://localhost:8000")
            
            self._llm = VLLMApi(
                vllm_host=vllm_host,
                model_name=self._model_name,
                #system_prompt="You are an expert at parsing task descriptions.",
                system_prompt="Ты эксперт по парсингу задач в структурированные коннекторы."
                #prompt="Ты эксперт по парсингу задач в структурированные коннекторы."
            )
            
            logger.info(f"VLLM backend initialized: {vllm_host}")
        
        elif self._backend == "deepseek":
            from ...tools.ApiTools.DeepSeekTool import DeepSeekApi
            
            self._llm = DeepSeekApi(
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                model_name=self._model_name or "deepseek-chat"
            )
            
            logger.info("DeepSeek backend initialized for NLI")
        
        else:
            raise ValueError(f"Unknown backend: {self._backend}")
    
    def load_dataset(self, examples: List[NLIDatasetItem]):
        """Загрузить датасет примеров для few-shot."""
        self._dataset = examples
        logger.info(f"Loaded {len(examples)} NLI examples")
    
    def parse_task(
        self,
        task_text: str,
        available_tools: List[BaseTool],
        k: int = 3
    ) -> NLIParseResult:
        """
        Распарсить задачу используя LLM с few-shot из k-NN.
        
        Алгоритм:
        1. Поиск k похожих примеров через k-NN
        2. Формирование few-shot промпта (по схеме из Qwen NLI)
        3. Вызов LLM
        4. Парсинг JSON ответа
        
        Args:
            task_text: Текст задачи на естественном языке
            available_tools: Доступные инструменты
            k: Количество примеров для few-shot
            
        Returns:
            NLIParseResult с коннекторами
        """
        if not task_text:
            return NLIParseResult(
                success=False,
                error_message="Task text cannot be empty"
            )
        
        # Шаг 1: k-NN поиск
        similar_examples = self._find_similar_examples(task_text, min(k, self._k_similar))
        
        # Шаг 2: Формирование промпта
        prompt = self._create_nli_prompt(task_text, similar_examples, available_tools)
        
        # Шаг 3: Вызов LLM
        try:
            response = self._call_llm(prompt)
            
            # Шаг 4: Парсинг ответа
            representation = self._parse_llm_response(response)
            
            if representation:
                # Шаг 5: Валидация — маппинг на доступные форматы
                representation = self._validate_connectors(
                    representation, available_tools
                )
                
                logger.info(
                    f"LLM NLI: '{task_text[:40]}...' -> "
                    f"{representation.input_connector.format} -> "
                    f"{representation.output_connector.format}"
                )
                
                return NLIParseResult(
                    success=True,
                    task_representation=representation,
                    similar_examples=similar_examples if similar_examples else None,
                    confidence=0.9
                )
            else:
                return NLIParseResult(
                    success=False,
                    error_message="Failed to parse LLM response"
                )
        
        except Exception as e:
            logger.error(f"Error in LLM NLI: {e}")
            return NLIParseResult(
                success=False,
                error_message=str(e)
            )
    
    def _find_similar_examples(self, task_text: str, k: int) -> List:
        """Поиск похожих примеров через k-NN."""
        if not self._dataset:
            return []
        
        query_emb = self._embedding_service.embed_text(task_text)
        similarities = []
        
        for example in self._dataset:
            if example.task_embedding:
                sim = self._embedding_service.compute_similarity(
                    query_emb, example.task_embedding
                )
                similarities.append((example, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, sim in similarities[:k]]
    
    def _create_nli_prompt(
        self,
        task_text: str,
        similar_examples: List[NLIDatasetItem],
        available_tools: List[BaseTool]
    ) -> str:
        """
        Создать промпт для LLM NLI.
        
        Воспроизводит структуру промпта файнтюненной Qwen NLI модели:
        - Описание коннекторов (dataType, semanticType, knowledgeDomain)
        - Допустимые значения
        - Few-shot примеры
        - Целевой запрос
        """
        # Собираем доступные типы и семантики ОТДЕЛЬНО для входа и выхода
        in_data_types = set()
        in_semantics = set()
        out_data_types = set()
        out_semantics = set()
        
        if available_tools:
            for t in available_tools:
                in_data_types.add(t.input.data_format)
                out_data_types.add(t.output.data_format)
                
                if t.input.semantic_format != "*":
                    in_semantics.add(t.input.semantic_format)
                if t.output.semantic_format != "*":
                    out_semantics.add(t.output.semantic_format)
                
                # Инструменты с * принимают/выдают ЛЮБУЮ семантику
                # для их data_type → добавляем все известные семантики
                if t.input.semantic_format == "*":
                    in_semantics.update(out_semantics)
                if t.output.semantic_format == "*":
                    out_semantics.update(in_semantics)
        
        # Финальные списки (без *)
        in_semantics.discard("*")
        out_semantics.discard("*")
        
        prompt = f"""Определи входной и выходной коннекторы для запроса.

Коннектор = subtype | semanticCategory

ВХОД (что принимает система):
  Типы данных: {sorted(in_data_types)}
  Семантика: {sorted(in_semantics)}

ВЫХОД (что выдает система):
  Типы данных: {sorted(out_data_types)}
  Семантика: {sorted(out_semantics)}

ПРАВИЛА:
- subtype и semanticCategory СТРОГО из списков выше
- Текстовый вопрос -> вход text|question
- Текстовый ответ -> выход text|answer
- "Нарисуй/схема/визуализируй" -> выход image|answer
- Если выход image, то semanticCategory ВСЕГДА = answer
- Если выход sound, то semanticCategory ВСЕГДА = answer

"""
        # Few-shot примеры (компактные)
        prompt += "ПРИМЕРЫ:\n"
        
        examples = []
        
        # Из k-NN
        if similar_examples:
            for ex in similar_examples[:3]:
                try:
                    rep = ex.representation
                    if isinstance(rep, dict):
                        ic = rep.get("input_connector", {})
                        oc = rep.get("output_connector", {})
                        examples.append((
                            ex.task_text,
                            ic.get("data_type", {}).get("subtype", "text"),
                            ic.get("semantic_type", {}).get("semantic_category", "question"),
                            oc.get("data_type", {}).get("subtype", "text"),
                            oc.get("semantic_type", {}).get("semantic_category", "answer"),
                        ))
                    elif rep and hasattr(rep, 'input_connector'):
                        ic, oc = rep.input_connector, rep.output_connector
                        if isinstance(ic, Connector):
                            examples.append((ex.task_text, ic.data_format, ic.semantic_format, oc.data_format, oc.semantic_format))
                except Exception:
                    pass
        
        # Базовые примеры из реальных типов (если нет k-NN)
        if not examples:
            if "text" in in_data_types:
                examples.append(("Ответь на вопрос", "text", "question", "text", "answer"))
            if "image" in out_data_types:
                examples.append(("Нарисуй схему", "text", "question", "image", "answer"))
            if "category" in out_semantics:
                examples.append(("Классифицируй текст", "text", "question", "text", "category"))
        
        for q, id, is_, od, os_ in examples:
            prompt += f'  "{q}" -> in:{id}|{is_} out:{od}|{os_}\n'
        
        # Целевой запрос
        prompt += f"""
Запрос: "{task_text}"
Выведи JSON: {{"input_connector":{{"dataType":{{"subtype":"?"}}, "semanticType":{{"semanticCategory":"?"}}}}, "output_connector":{{"dataType":{{"subtype":"?"}}, "semanticType":{{"semanticCategory":"?"}}}}}}"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Вызвать LLM."""
        if self._backend in ["openrouter", "vllm", "deepseek"]:
            response = self._llm.query_llm(
                question=prompt,
                temperature=self._temperature,
                max_tokens=500
            )
            return response
        else:
            raise ValueError(f"Unknown backend: {self._backend}")
    
    def _parse_llm_response(self, response: str) -> Optional[TaskRepresentation]:
        """
        Распарсить JSON ответ от LLM.
        
        Поддерживает два формата:
        1. Новый (из Qwen NLI): dataType/semanticType/knowledgeDomain
        2. Старый: data_format/semantic_format
        """
        try:
            # Ищем JSON в ответе
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in LLM response")
                logger.debug(f"Response: {response[:200]}")
                return None
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Парсинг входного коннектора
            input_conn = self._parse_connector_from_json(data.get("input_connector", {}))
            output_conn = self._parse_connector_from_json(data.get("output_connector", {}))
            
            if not input_conn or not output_conn:
                logger.error("Failed to extract connectors from JSON")
                return None
            
            # Создание представления
            representation = TaskRepresentation()
            representation.input_connector = input_conn
            representation.output_connector = output_conn
            
            return representation
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response: {response[:200]}")
            return None
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _parse_connector_from_json(self, conn_data: Dict) -> Optional[Connector]:
        """
        Парсинг коннектора из JSON.
        
        Поддерживает оба формата:
        - Новый: {"dataType": {"complexType": ..., "subtype": ...}, "semanticType": {"semanticCategory": ...}}
        - Старый: {"data_format": "...", "semantic_format": "..."}
        """
        if not conn_data:
            return None
        
        # Новый формат (из Qwen NLI)
        if "dataType" in conn_data:
            dt = conn_data["dataType"]
            st = conn_data.get("semanticType", {})
            
            # subtype определяет формат данных
            data_format = dt.get("subtype", "text")
            semantic_format = st.get("semanticCategory", "question")
            
            return Connector(data_format, semantic_format)
        
        # Старый формат (простой)
        if "data_format" in conn_data:
            return Connector(
                conn_data["data_format"],
                conn_data.get("semantic_format", "data")
            )
        
        logger.warning(f"Unknown connector format: {conn_data}")
        return None
    
    def _validate_connectors(
        self,
        representation: TaskRepresentation,
        available_tools: List[BaseTool]
    ) -> TaskRepresentation:
        """
        Валидация коннекторов: если формат не существует среди инструментов,
        заменяем на ближайший доступный.
        Все данные берутся из реальных инструментов, без хардкода.
        """
        if not available_tools:
            return representation
        
        # Собираем доступные значения из инструментов (раскрываем * в конкретные)
        valid_subtypes = set()
        valid_semantics = set()
        valid_input_formats = set()
        valid_output_formats = set()
        
        # Сначала собираем конкретные семантики
        for t in available_tools:
            valid_subtypes.add(t.input.data_format)
            valid_subtypes.add(t.output.data_format)
            if t.input.semantic_format != "*":
                valid_semantics.add(t.input.semantic_format)
            if t.output.semantic_format != "*":
                valid_semantics.add(t.output.semantic_format)
        
        # Раскрываем * форматы
        for t in available_tools:
            if t.input.semantic_format == "*":
                for sem in valid_semantics:
                    valid_input_formats.add(f"{t.input.data_format}|{sem}")
            else:
                valid_input_formats.add(t.input.format)
            
            if t.output.semantic_format == "*":
                for sem in valid_semantics:
                    valid_output_formats.add(f"{t.output.data_format}|{sem}")
            else:
                valid_output_formats.add(t.output.format)
        
        def find_closest(val: str, valid_set: set) -> str:
            """Найти ближайшее значение из допустимого множества."""
            if val in valid_set:
                return val
            
            # Поиск по вхождению подстроки
            val_lower = val.lower()
            for v in valid_set:
                if val_lower in v.lower() or v.lower() in val_lower:
                    logger.info(f"NLI validate: '{val}' -> '{v}' (substring match)")
                    return v
            
            # Ничего не нашли
            return None
        
        def fix_connector(conn, is_input: bool) -> Connector:
            """Исправить коннектор если его формат невалиден."""
            if not isinstance(conn, Connector):
                return conn
            
            valid_formats = valid_input_formats if is_input else valid_output_formats
            
            # Если формат уже валиден — ничего не делаем
            if conn.format in valid_formats:
                return conn
            
            # Пробуем исправить subtype
            new_dt = find_closest(conn.data_format, valid_subtypes)
            new_sem = find_closest(conn.semantic_format, valid_semantics)
            
            if not new_dt:
                new_dt = "text"
                logger.warning(f"NLI validate: unknown subtype '{conn.data_format}', fallback 'text'")
            if not new_sem:
                new_sem = "question" if is_input else "answer"
                logger.warning(f"NLI validate: unknown semantic '{conn.semantic_format}', fallback '{new_sem}'")
            
            fixed = Connector(new_dt, new_sem)
            
            # Проверяем что исправленный формат реально существует
            if fixed.format not in valid_formats:
                # Берем самый частый формат
                default = "text|question" if is_input else "text|answer"
                parts = default.split("|")
                fixed = Connector(parts[0], parts[1])
                logger.warning(f"NLI validate: '{conn.format}' -> '{fixed.format}' (forced default)")
            else:
                logger.info(f"NLI validate: '{conn.format}' -> '{fixed.format}'")
            
            return fixed
        
        # Семантические правила: нетекстовые типы данных имеют ограниченные семантики
        # image и sound на выходе — всегда answer (изображение/звук = ответ на запрос)
        out_conn = representation.output_connector
        if isinstance(out_conn, Connector) and out_conn.data_format in ("image", "sound"):
            if out_conn.semantic_format != "answer":
                logger.info(f"NLI rule: {out_conn.data_format}|{out_conn.semantic_format} -> {out_conn.data_format}|answer")
                representation.output_connector = Connector(out_conn.data_format, "answer")
        
        representation.input_connector = fix_connector(
            representation.input_connector, is_input=True
        )
        representation.output_connector = fix_connector(
            representation.output_connector, is_input=False
        )
        
        return representation
    
    def is_available(self) -> bool:
        """Проверить доступность LLM."""
        return self._llm is not None
