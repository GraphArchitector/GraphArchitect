"""
Фабрика для создания NLI сервисов.

Автоматически выбирает подходящий NLI на основе конфигурации:
- k-NN few-shot (базовый, всегда работает)
- Qwen fine-tuned (если есть модель)
- LLM-based (если есть API ключ или VLLM)
"""

import logging
from typing import Optional
import os

from ..embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def create_nli_service(
    embedding_service: EmbeddingService,
    nli_type: str = "llm",  # "knn", "qwen", "llm"
    llm_backend: str = "openrouter",
    llm_model: str = "openai/gpt-3.5-turbo",
    qwen_model_path: Optional[str] = None,
    vllm_host: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Создать NLI сервис на основе конфигурации.
    
    Args:
        embedding_service: Сервис эмбеддингов
        nli_type: Тип NLI ("knn", "qwen", "llm")
        llm_backend: Бэкенд для LLM NLI
        llm_model: Модель для LLM NLI
        qwen_model_path: Путь к fine-tuned Qwen
        vllm_host: URL VLLM сервера
        api_key: API ключ
        
    Returns:
        Инициализированный NLI сервис
    """
    
    if nli_type == "llm":
        logger.info("Creating LLM-based NLI...")
        
        try:
            from .llm_nli_service import LLMNLIService
            
            # Проверка наличия API ключа
            if llm_backend == "openrouter" and not (api_key or os.getenv("OPENROUTER_API_KEY")):
                logger.warning("OpenRouter API key not found, falling back to k-NN NLI")
                nli_type = "knn"
            elif llm_backend == "deepseek" and not (api_key or os.getenv("DEEPSEEK_API_KEY")):
                logger.warning("DeepSeek API key not found, falling back to k-NN NLI")
                nli_type = "knn"
            else:
                nli = LLMNLIService(
                    embedding_service=embedding_service,
                    backend=llm_backend,
                    model_name=llm_model,
                    api_key=api_key,
                    vllm_host=vllm_host
                )
                
                if nli.is_available():
                    logger.info(f"LLM NLI created: {llm_backend} with {llm_model}")
                    return nli
                else:
                    logger.warning("LLM not available, falling back to k-NN")
                    nli_type = "knn"
        
        except Exception as e:
            logger.error(f"Failed to create LLM NLI: {e}")
            logger.info("Falling back to k-NN NLI")
            nli_type = "knn"
    
    if nli_type == "qwen":
        logger.info("Creating Qwen fine-tuned NLI...")
        
        try:
            from .qwen_nli_service import QwenNLIService
            
            if not qwen_model_path:
                logger.warning("Qwen model path not provided, falling back to k-NN")
                nli_type = "knn"
            else:
                nli = QwenNLIService(model_path=qwen_model_path)
                
                if nli.is_available():
                    logger.info(f"Qwen NLI created from {qwen_model_path}")
                    return nli
                else:
                    logger.warning("Qwen model not available, falling back to k-NN")
                    nli_type = "knn"
        
        except Exception as e:
            logger.error(f"Failed to create Qwen NLI: {e}")
            logger.info("Falling back to k-NN NLI")
            nli_type = "knn"
    
    # Fallback: k-NN (всегда работает)
    if nli_type == "knn" or True:  # Всегда доступен как fallback
        logger.info("Creating k-NN few-shot NLI...")
        
        from .natural_language_interface import NaturalLanguageInterface
        
        nli = NaturalLanguageInterface(embedding_service)
        logger.info("k-NN NLI created (always available)")
        return nli


def create_nli_service_from_env(embedding_service: EmbeddingService):
    """
    Создать NLI из переменных окружения.
    
    Переменные:
    - NLI_TYPE: "knn", "qwen", "llm"
    - NLI_LLM_BACKEND: "openrouter", "vllm", "deepseek"
    - NLI_LLM_MODEL: название модели
    - QWEN_MODEL_PATH: путь к Qwen модели
    - VLLM_HOST: URL VLLM
    
    Args:
        embedding_service: Сервис эмбеддингов
        
    Returns:
        NLI сервис
    """
    return create_nli_service(
        embedding_service=embedding_service,
        nli_type=os.getenv("NLI_TYPE", "llm"),
        llm_backend=os.getenv("NLI_LLM_BACKEND", "openrouter"),
        llm_model=os.getenv("NLI_LLM_MODEL", "openai/gpt-3.5-turbo"),
        qwen_model_path=os.getenv("QWEN_MODEL_PATH"),
        vllm_host=os.getenv("VLLM_HOST")
    )
