"""
OpenRouter API клиент для GraphArchitect.

OpenRouter - универсальный роутер для доступа к различным LLM:
- OpenAI, Anthropic, Google, Meta, Mistral и др.
- Один API ключ для всех моделей
- Автоматический failover

Документация: https://openrouter.ai/docs
"""

import requests
import logging
from typing import Any, Dict, Optional, List
import os

logger = logging.getLogger(__name__)


class OpenRouterLLM:
    """
    Клиент для взаимодействия с OpenRouter API.
    
    OpenRouter предоставляет доступ к множеству LLM через единый интерфейс.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-3.5-turbo",
        system_prompt: str = "You are a helpful AI assistant.",
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "GraphArchitect",
        app_url: str = "https://github.com/FractalAgentsAI/GraphArchitect"
    ):
        """
        Инициализация OpenRouter клиента.
        
        Args:
            api_key: API ключ OpenRouter (или OPENROUTER_API_KEY из env)
            model_name: ID модели (см. https://openrouter.ai/models)
                Примеры:
                - openai/gpt-4
                - openai/gpt-3.5-turbo
                - anthropic/claude-3.5-sonnet
                - google/gemini-pro
                - meta-llama/llama-3-70b
                - mistralai/mistral-large
            system_prompt: Системный промпт
            base_url: Базовый URL API
            app_name: Имя приложения (для статистики)
            app_url: URL приложения
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API ключ не указан. "
                "Установите переменную окружения OPENROUTER_API_KEY "
                "или передайте api_key в конструктор."
            )
        
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.endpoint = f"{base_url}/chat/completions"
        
        # Заголовки для OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": app_url,  # Опционально, для статистики
            "X-Title": app_name        # Опционально, для статистики
        }
        
        logger.info(f"OpenRouter инициализирован: модель {model_name}")
    
    def query_llm(
        self,
        question: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1000,
        timeout: int = 30,
        **kwargs
    ) -> str:
        """
        Отправить запрос к LLM через OpenRouter.
        
        Args:
            question: Вопрос пользователя
            temperature: Параметр температуры (0.0-2.0)
            top_p: Nucleus sampling
            max_tokens: Максимум токенов в ответе
            timeout: Таймаут в секундах
            **kwargs: Дополнительные параметры (presence_penalty, frequency_penalty и др.)
        
        Returns:
            Ответ модели или сообщение об ошибке
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        # Добавляем дополнительные параметры
        payload.update(kwargs)
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Извлекаем ответ
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                logger.warning(f"Пустой ответ от модели {self.model_name}")
                return "Empty response from model"
            
            # Логируем использование токенов (опционально)
            usage = data.get("usage", {})
            if usage:
                logger.info(
                    f"Использовано токенов: "
                    f"prompt={usage.get('prompt_tokens')}, "
                    f"completion={usage.get('completion_tokens')}, "
                    f"total={usage.get('total_tokens')}"
                )
            
            return content
        
        except requests.exceptions.Timeout:
            logger.error(f"Таймаут при запросе к {self.model_name}")
            return f"Request timeout after {timeout}s"
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP ошибка: {e.response.status_code} - {e.response.text}")
            return f"HTTP error: {e.response.status_code}"
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к OpenRouter: {e}")
            return f"Request failed: {e}"
        
        except (KeyError, IndexError) as e:
            logger.error(f"Неожиданный формат ответа: {e}")
            return f"Unexpected response format: {e}"
    
    def query_with_context(
        self,
        question: str,
        context: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30
    ) -> str:
        """
        Запрос с контекстом (история сообщений).
        
        Args:
            question: Текущий вопрос
            context: История сообщений [{"role": "user/assistant", "content": "..."}]
            temperature: Температура
            max_tokens: Максимум токенов
            timeout: Таймаут
        
        Returns:
            Ответ модели
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(context)
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return content or "Empty response"
        
        except Exception as e:
            logger.error(f"Ошибка в query_with_context: {e}")
            return f"Error: {e}"
    
    def get_available_models(self, timeout: int = 10) -> List[Dict[str, Any]]:
        """
        Получить список доступных моделей.
        
        Args:
            timeout: Таймаут запроса
        
        Returns:
            Список моделей с их параметрами
        """
        models_endpoint = f"{self.base_url}/models"
        
        try:
            response = requests.get(
                models_endpoint,
                headers=self.headers,
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            models = data.get("data", [])
            
            logger.info(f"Доступно моделей: {len(models)}")
            
            return models
        
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            return []


class OpenRouterTool:
    """
    Обертка OpenRouterLLM для использования в GraphArchitect.
    
    Это адаптер между OpenRouter API и BaseTool интерфейсом.
    Наследуйте от BaseTool и используйте этот класс внутри execute().
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-3.5-turbo",
        system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Инициализация OpenRouter инструмента.
        
        Args:
            api_key: OpenRouter API ключ
            model_name: ID модели
            system_prompt: Системный промпт
        """
        self.client = OpenRouterLLM(
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt
        )
        self.model_name = model_name
    
    def execute(self, input_data: str, **kwargs) -> str:
        """
        Выполнить запрос к LLM.
        
        Args:
            input_data: Входной текст/вопрос
            **kwargs: Дополнительные параметры (temperature, max_tokens и др.)
        
        Returns:
            Ответ от LLM
        """
        return self.client.query_llm(input_data, **kwargs)
    
    def execute_with_context(
        self,
        question: str,
        context: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Выполнить запрос с контекстом.
        
        Args:
            question: Вопрос
            context: История диалога
            **kwargs: Параметры
        
        Returns:
            Ответ от LLM
        """
        return self.client.query_with_context(question, context, **kwargs)
