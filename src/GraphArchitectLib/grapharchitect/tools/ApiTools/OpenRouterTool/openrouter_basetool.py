"""
Реализации BaseTool с OpenRouter для использования в GraphArchitect.

Готовые инструменты для разных задач:
- OpenRouterChatTool - общий чат
- OpenRouterClassifierTool - классификация
- OpenRouterSummarizerTool - суммаризация
- OpenRouterAnalyzerTool - анализ
"""

from typing import Optional, List, Dict

# Импорты с относительными путями
try:
    # Пробуем импортировать напрямую (когда grapharchitect установлен)
    from grapharchitect.entities.base_tool import BaseTool
    from grapharchitect.entities.connectors.connector import Connector
except ImportError:
    # Fallback для разработки
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from grapharchitect.entities.base_tool import BaseTool
    from grapharchitect.entities.connectors.connector import Connector

from .openrouter_llm import OpenRouterLLM
from .openrouter_config import OpenRouterConfig


class OpenRouterChatTool(BaseTool):
    """
    Универсальный чат-инструмент с OpenRouter.
    
    Может использовать любую модель через OpenRouter API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_key: str = "gpt-3.5-turbo",
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7
    ):
        """
        Инициализация чат-инструмента.
        
        Args:
            api_key: OpenRouter API ключ
            model_key: Ключ модели из OpenRouterConfig.MODELS
            system_prompt: Системный промпт
            temperature: Температура генерации
        """
        super().__init__()
        
        # Получаем конфигурацию модели
        model_config = OpenRouterConfig.get_model(model_key)
        
        if model_config:
            model_id = model_config.model_id
            self.metadata.tool_name = f"OpenRouter-{model_config.display_name}"
            self.metadata.description = f"Chat using {model_config.display_name}"
            self.metadata.mean_cost = model_config.cost_per_1m_tokens / 1000  # На 1K токенов
            self.temperature = model_config.recommended_temperature
        else:
            model_id = model_key
            self.metadata.tool_name = f"OpenRouter-{model_key}"
            self.metadata.description = f"Chat using {model_key}"
            self.temperature = temperature
        
        # Коннекторы - универсальные для чата
        self.input = Connector("text", "question")
        self.output = Connector("text", "answer")
        
        # Создаем клиент OpenRouter
        self.client = OpenRouterLLM(
            api_key=api_key,
            model_name=model_id,
            system_prompt=system_prompt
        )
        
        self.model_key = model_key
    
    def execute(self, input_data) -> str:
        """
        Выполнить запрос к LLM.
        
        Args:
            input_data: Вопрос пользователя
        
        Returns:
            Ответ от модели
        """
        return self.client.query_llm(
            str(input_data),
            temperature=self.temperature
        )


class OpenRouterClassifierTool(BaseTool):
    """
    Инструмент классификации с OpenRouter.
    
    Классифицирует текст в одну из заданных категорий.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_key: str = "gpt-3.5-turbo",
        categories: Optional[list] = None
    ):
        """
        Инициализация классификатора.
        
        Args:
            api_key: OpenRouter API ключ
            model_key: Ключ модели
            categories: Список категорий для классификации
        """
        super().__init__()
        
        self.categories = categories or ["positive", "negative", "neutral"]
        
        # Получаем конфигурацию
        model_config = OpenRouterConfig.get_model(model_key)
        model_id = model_config.model_id if model_config else model_key
        
        self.metadata.tool_name = f"Classifier-{model_key}"
        self.metadata.description = f"Классификация текста с {model_key}"
        
        # Коннекторы
        self.input = Connector("text", "question")
        self.output = Connector("text", "category")
        
        # Системный промпт для классификации
        categories_str = ", ".join(self.categories)
        system_prompt = (
            f"You are a text classifier. "
            f"Classify the given text into one of these categories: {categories_str}. "
            f"Respond with ONLY the category name, nothing else."
        )
        
        self.client = OpenRouterLLM(
            api_key=api_key,
            model_name=model_id,
            system_prompt=system_prompt
        )
    
    def execute(self, input_data) -> str:
        """
        Классифицировать текст.
        
        Args:
            input_data: Текст для классификации
        
        Returns:
            Категория
        """
        result = self.client.query_llm(str(input_data), temperature=0.2)
        
        # Очищаем результат (только категория)
        result = result.strip().lower()
        
        # Проверяем что результат - одна из категорий
        for category in self.categories:
            if category.lower() in result:
                return category
        
        return result


class OpenRouterSummarizerTool(BaseTool):
    """
    Инструмент суммаризации с OpenRouter.
    
    Создает краткую сводку из длинного текста.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_key: str = "gpt-3.5-turbo",
        max_summary_words: int = 100
    ):
        """
        Инициализация суммаризатора.
        
        Args:
            api_key: OpenRouter API ключ
            model_key: Ключ модели
            max_summary_words: Максимум слов в сводке
        """
        super().__init__()
        
        model_config = OpenRouterConfig.get_model(model_key)
        model_id = model_config.model_id if model_config else model_key
        
        self.metadata.tool_name = f"Summarizer-{model_key}"
        self.metadata.description = f"Суммаризация с {model_key}"
        
        # Коннекторы
        self.input = Connector("text", "document")
        self.output = Connector("text", "summary")
        
        system_prompt = (
            f"You are a text summarizer. "
            f"Create a concise summary of the given text in no more than {max_summary_words} words. "
            f"Focus on the main points and key information."
        )
        
        self.client = OpenRouterLLM(
            api_key=api_key,
            model_name=model_id,
            system_prompt=system_prompt
        )
    
    def execute(self, input_data) -> str:
        """
        Создать сводку текста.
        
        Args:
            input_data: Текст для суммаризации
        
        Returns:
            Краткая сводка
        """
        return self.client.query_llm(str(input_data), temperature=0.3)


class OpenRouterAnalyzerTool(BaseTool):
    """
    Инструмент анализа с OpenRouter.
    
    Анализирует текст и предоставляет инсайты.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_key: str = "gpt-4",
        analysis_type: str = "general"
    ):
        """
        Инициализация анализатора.
        
        Args:
            api_key: OpenRouter API ключ
            model_key: Ключ модели
            analysis_type: Тип анализа (general, sentiment, technical и т.д.)
        """
        super().__init__()
        
        model_config = OpenRouterConfig.get_model(model_key)
        model_id = model_config.model_id if model_config else model_key
        
        self.metadata.tool_name = f"Analyzer-{model_key}"
        self.metadata.description = f"Анализ с {model_key}"
        
        # Коннекторы
        self.input = Connector("text", "data")
        self.output = Connector("text", "analysis")
        
        # Системный промпт в зависимости от типа анализа
        prompts = {
            "general": "You are a general text analyzer. Provide detailed analysis of the given text.",
            "sentiment": "You are a sentiment analyzer. Analyze the sentiment and emotions in the text.",
            "technical": "You are a technical analyzer. Analyze technical aspects and provide expert insights."
        }
        
        system_prompt = prompts.get(analysis_type, prompts["general"])
        
        self.client = OpenRouterLLM(
            api_key=api_key,
            model_name=model_id,
            system_prompt=system_prompt
        )
    
    def execute(self, input_data) -> str:
        """
        Проанализировать текст.
        
        Args:
            input_data: Текст для анализа
        
        Returns:
            Результат анализа
        """
        return self.client.query_llm(str(input_data), temperature=0.5)
