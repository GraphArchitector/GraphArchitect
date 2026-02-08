"""
Сервис NLI на основе дообученной Qwen модели.

Реализует подход из отчета:
- Использование дообученной Qwen 2.5 7b
- Few-shot промпт (2 примера)
- Структурное описание выхода
- Вывод коннекторов в формате JSON
"""

import logging
from typing import List, Optional, Dict, Any
import json

from ...entities.base_tool import BaseTool
from ...entities.connectors.task_representation import TaskRepresentation
from ...entities.connectors.connector import Connector
from ..embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class QwenNLIService:
    """
    NLI на основе дообученной Qwen модели.
    
    Использует fine-tuned Qwen 2.5 7b для преобразования
    естественного языка в пару коннекторов.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        Инициализация сервиса.
        
        Args:
            model_path: Путь к дообученной модели Qwen
            device: Устройство (cpu, cuda)
            max_length: Максимальная длина токенов
        """
        self._model_path = model_path
        self._device = device
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        
        try:
            self._load_model()
            logger.info(f"QwenNLIService initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            logger.warning("QwenNLIService will not be available")
    
    def _load_model(self):
        """Загрузить дообученную Qwen модель."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map=self._device
            )
            
            self._model.eval()
            
        except ImportError as e:
            logger.error(f"Transformers not installed: {e}")
            raise
    
    def parse_task(
        self,
        task_text: str,
        available_tools: List[BaseTool],
        temperature: float = 0.1
    ) -> Optional[TaskRepresentation]:
        """
        Распарсить задачу используя Qwen модель.
        
        Args:
            task_text: Текст задачи на естественном языке
            available_tools: Доступные инструменты (для контекста)
            temperature: Температура генерации
            
        Returns:
            TaskRepresentation с коннекторами или None
        """
        if not self._model:
            logger.error("Qwen model not loaded")
            return None
        
        # Формирование промпта (из отчета)
        prompt = self._create_prompt(task_text, available_tools)
        
        try:
            # Генерация
            inputs = self._tokenizer(prompt, return_tensors="pt")
            
            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Декодирование
            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлечение JSON с коннекторами
            representation = self._parse_generated_json(generated_text)
            
            if representation:
                logger.info(
                    f"Qwen NLI parsed: {representation.input_connector.format} → "
                    f"{representation.output_connector.format}"
                )
            
            return representation
        
        except Exception as e:
            logger.error(f"Error in Qwen generation: {e}")
            return None
    
    def _create_prompt(
        self,
        task_text: str,
        available_tools: List[BaseTool]
    ) -> str:
        """
        Создать промпт для Qwen (из отчета).
        
        Включает:
        - Описание форматов коннекторов
        - Few-shot примеры (2 шт)
        - Структуру выхода
        - Задачу пользователя
        """
        # Описание доступных коннекторов
        available_formats = self._get_available_formats(available_tools)
        
        prompt = f"""Задача: Преобразовать описание задачи в формат входного и выходного коннектора.

Доступные форматы данных:
{available_formats}

Примеры (few-shot):

Пример 1:
Задача: "Классифицировать текст по категориям"
Выход:
{{
  "input_connector": {{
    "data_format": "text",
    "semantic_format": "question"
  }},
  "output_connector": {{
    "data_format": "text",
    "semantic_format": "category"
  }}
}}

Пример 2:
Задача: "Ответить на вопрос пользователя"
Выход:
{{
  "input_connector": {{
    "data_format": "text",
    "semantic_format": "question"
  }},
  "output_connector": {{
    "data_format": "text",
    "semantic_format": "answer"
  }}
}}

Теперь ваша задача:
Задача: "{task_text}"
Выход:
"""
        
        return prompt
    
    def _get_available_formats(self, tools: List[BaseTool]) -> str:
        """Получить список доступных форматов из инструментов."""
        formats = set()
        
        for tool in tools:
            formats.add(tool.input.format)
            formats.add(tool.output.format)
        
        return "\n".join(f"  - {fmt}" for fmt in sorted(formats))
    
    def _parse_generated_json(self, generated_text: str) -> Optional[TaskRepresentation]:
        """
        Извлечь JSON с коннекторами из сгенерированного текста.
        
        Args:
            generated_text: Сгенерированный текст от Qwen
            
        Returns:
            TaskRepresentation или None
        """
        try:
            # Ищем JSON в тексте
            start_idx = generated_text.find('{')
            end_idx = generated_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in generated text")
                return None
            
            json_str = generated_text[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Создаем коннекторы
            input_conn = Connector(
                data['input_connector']['data_format'],
                data['input_connector']['semantic_format']
            )
            
            output_conn = Connector(
                data['output_connector']['data_format'],
                data['output_connector']['semantic_format']
            )
            
            # Создаем представление
            representation = TaskRepresentation()
            representation.input_connector = input_conn
            representation.output_connector = output_conn
            
            return representation
        
        except Exception as e:
            logger.error(f"Failed to parse JSON from Qwen output: {e}")
            logger.debug(f"Generated text: {generated_text}")
            return None
    
    def is_available(self) -> bool:
        """
        Проверить доступность модели.
        
        Returns:
            True если модель загружена
        """
        return self._model is not None
