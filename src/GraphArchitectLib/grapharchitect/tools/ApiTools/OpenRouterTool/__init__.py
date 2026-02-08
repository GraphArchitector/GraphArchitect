"""
OpenRouter API Tools - универсальный доступ к множеству LLM.

OpenRouter предоставляет единый API для:
- OpenAI (GPT-4, GPT-3.5-turbo)
- Anthropic (Claude 3.5, Claude 3)
- Google (Gemini Pro, Gemini Ultra)
- Meta (Llama 3, Llama 2)
- Mistral (Mistral Large, Medium)
- И многие другие модели

Преимущества:
- Один API ключ для всех моделей
- Автоматический failover
- Прозрачная балансировка
- Конкурентные цены
"""

from .openrouter_llm import OpenRouterLLM, OpenRouterTool
from .openrouter_config import OpenRouterConfig, ModelConfig

__all__ = [
    'OpenRouterLLM',
    'OpenRouterTool',
    'OpenRouterConfig',
    'ModelConfig'
]
