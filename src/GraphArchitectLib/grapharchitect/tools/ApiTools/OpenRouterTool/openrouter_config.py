"""
Конфигурация для OpenRouter моделей.

Содержит популярные модели и их параметры.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Конфигурация модели OpenRouter"""
    
    # ID модели в OpenRouter
    model_id: str
    
    # Человекочитаемое имя
    display_name: str
    
    # Провайдер (openai, anthropic, google и т.д.)
    provider: str
    
    # Ориентировочная стоимость за 1M токенов (USD)
    cost_per_1m_tokens: float = 0.0
    
    # Максимальное количество токенов контекста
    max_context_tokens: int = 4096
    
    # Максимальное количество токенов в ответе
    max_output_tokens: int = 4096
    
    # Рекомендуемая температура
    recommended_temperature: float = 0.7
    
    # Поддерживает ли function calling
    supports_functions: bool = False
    
    # Поддерживает ли vision
    supports_vision: bool = False
    
    # Дополнительные параметры
    metadata: Dict[str, Any] = field(default_factory=dict)


class OpenRouterConfig:
    """
    Конфигурация популярных моделей OpenRouter.
    
    Содержит предустановленные настройки для разных моделей.
    """
    
    # Популярные модели
    MODELS = {
        # ===== OpenAI =====
        "gpt-4-turbo": ModelConfig(
            model_id="openai/gpt-4-turbo",
            display_name="GPT-4 Turbo",
            provider="openai",
            cost_per_1m_tokens=10.0,
            max_context_tokens=128000,
            max_output_tokens=4096,
            recommended_temperature=0.7,
            supports_functions=True,
            supports_vision=True
        ),
        
        "gpt-4": ModelConfig(
            model_id="openai/gpt-4",
            display_name="GPT-4",
            provider="openai",
            cost_per_1m_tokens=30.0,
            max_context_tokens=8192,
            max_output_tokens=8192,
            recommended_temperature=0.7,
            supports_functions=True
        ),
        
        "gpt-3.5-turbo": ModelConfig(
            model_id="openai/gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            provider="openai",
            cost_per_1m_tokens=0.5,
            max_context_tokens=16384,
            max_output_tokens=4096,
            recommended_temperature=0.7,
            supports_functions=True
        ),
        
        # ===== Anthropic =====
        "claude-3.5-sonnet": ModelConfig(
            model_id="anthropic/claude-3.5-sonnet",
            display_name="Claude 3.5 Sonnet",
            provider="anthropic",
            cost_per_1m_tokens=3.0,
            max_context_tokens=200000,
            max_output_tokens=4096,
            recommended_temperature=0.7,
            supports_vision=True
        ),
        
        "claude-3-opus": ModelConfig(
            model_id="anthropic/claude-3-opus",
            display_name="Claude 3 Opus",
            provider="anthropic",
            cost_per_1m_tokens=15.0,
            max_context_tokens=200000,
            max_output_tokens=4096,
            recommended_temperature=0.7,
            supports_vision=True
        ),
        
        "claude-3-sonnet": ModelConfig(
            model_id="anthropic/claude-3-sonnet",
            display_name="Claude 3 Sonnet",
            provider="anthropic",
            cost_per_1m_tokens=3.0,
            max_context_tokens=200000,
            max_output_tokens=4096,
            recommended_temperature=0.7
        ),
        
        # ===== Google =====
        "gemini-pro": ModelConfig(
            model_id="google/gemini-pro",
            display_name="Gemini Pro",
            provider="google",
            cost_per_1m_tokens=0.125,
            max_context_tokens=32000,
            max_output_tokens=2048,
            recommended_temperature=0.7
        ),
        
        "gemini-pro-vision": ModelConfig(
            model_id="google/gemini-pro-vision",
            display_name="Gemini Pro Vision",
            provider="google",
            cost_per_1m_tokens=0.125,
            max_context_tokens=16000,
            max_output_tokens=2048,
            recommended_temperature=0.7,
            supports_vision=True
        ),
        
        # ===== Meta Llama =====
        "llama-3-70b": ModelConfig(
            model_id="meta-llama/llama-3-70b-instruct",
            display_name="Llama 3 70B",
            provider="meta",
            cost_per_1m_tokens=0.52,
            max_context_tokens=8192,
            max_output_tokens=8192,
            recommended_temperature=0.7
        ),
        
        "llama-3-8b": ModelConfig(
            model_id="meta-llama/llama-3-8b-instruct",
            display_name="Llama 3 8B",
            provider="meta",
            cost_per_1m_tokens=0.06,
            max_context_tokens=8192,
            max_output_tokens=8192,
            recommended_temperature=0.7
        ),
        
        # ===== Mistral =====
        "mistral-large": ModelConfig(
            model_id="mistralai/mistral-large",
            display_name="Mistral Large",
            provider="mistral",
            cost_per_1m_tokens=3.0,
            max_context_tokens=32000,
            max_output_tokens=8192,
            recommended_temperature=0.7
        ),
        
        "mixtral-8x7b": ModelConfig(
            model_id="mistralai/mixtral-8x7b-instruct",
            display_name="Mixtral 8x7B",
            provider="mistral",
            cost_per_1m_tokens=0.24,
            max_context_tokens=32000,
            max_output_tokens=8192,
            recommended_temperature=0.7
        ),
        
        # ===== DeepSeek =====
        "deepseek-chat": ModelConfig(
            model_id="deepseek/deepseek-chat",
            display_name="DeepSeek Chat",
            provider="deepseek",
            cost_per_1m_tokens=0.14,
            max_context_tokens=64000,
            max_output_tokens=4096,
            recommended_temperature=0.7
        ),
        
        # ===== Free модели (для тестирования) =====
        "free-gpt-3.5": ModelConfig(
            model_id="openai/gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo (Free)",
            provider="openai",
            cost_per_1m_tokens=0.0,  # Лимитированный бесплатный доступ
            max_context_tokens=16384,
            max_output_tokens=4096,
            recommended_temperature=0.7
        ),
    }
    
    @classmethod
    def get_model(cls, model_key: str) -> Optional[ModelConfig]:
        """
        Получить конфигурацию модели по ключу.
        
        Args:
            model_key: Ключ модели (например "gpt-4", "claude-3.5-sonnet")
        
        Returns:
            ModelConfig или None если не найдено
        """
        return cls.MODELS.get(model_key)
    
    @classmethod
    def get_model_id(cls, model_key: str) -> str:
        """
        Получить ID модели для OpenRouter API.
        
        Args:
            model_key: Ключ модели
        
        Returns:
            ID модели (например "openai/gpt-4")
        """
        model = cls.get_model(model_key)
        return model.model_id if model else model_key
    
    @classmethod
    def list_models(cls) -> Dict[str, ModelConfig]:
        """Получить все доступные модели"""
        return cls.MODELS.copy()
    
    @classmethod
    def list_by_provider(cls, provider: str) -> Dict[str, ModelConfig]:
        """
        Получить модели конкретного провайдера.
        
        Args:
            provider: Имя провайдера (openai, anthropic, google и т.д.)
        
        Returns:
            Словарь моделей этого провайдера
        """
        return {
            key: model
            for key, model in cls.MODELS.items()
            if model.provider == provider
        }
    
    @classmethod
    def get_cheapest_model(cls) -> ModelConfig:
        """Получить самую дешевую модель"""
        return min(cls.MODELS.values(), key=lambda m: m.cost_per_1m_tokens)
    
    @classmethod
    def get_best_model(cls) -> ModelConfig:
        """Получить лучшую модель (по умолчанию GPT-4)"""
        return cls.MODELS.get("gpt-4-turbo") or list(cls.MODELS.values())[0]
