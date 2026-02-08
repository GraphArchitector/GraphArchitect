"""
Фабрика для создания сервисов эмбеддингов.

Автоматически выбирает подходящий EmbeddingService на основе конфигурации:
- SimpleEmbeddingService - заглушка (хеширование)
- InfinityEmbeddingService - реальные эмбеддинги через Infinity API
"""

import logging
from typing import Optional

from .embedding_service import EmbeddingService
from .simple_embedding_service import SimpleEmbeddingService

logger = logging.getLogger(__name__)


def create_embedding_service(
    embedding_type: str = "simple",
    dimension: int = 384,
    infinity_url: Optional[str] = None,
    infinity_api_key: Optional[str] = None,
    infinity_model: str = "BAAI/bge-small-en-v1.5",
    infinity_timeout: int = 10,
    fallback_to_simple: bool = True
) -> EmbeddingService:
    """
    Создать сервис эмбеддингов на основе конфигурации.
    
    Args:
        embedding_type: Тип сервиса ("simple", "infinity")
        dimension: Размерность эмбеддингов
        infinity_url: URL Infinity сервера
        infinity_api_key: API ключ для Infinity
        infinity_model: Модель эмбеддингов
        infinity_timeout: Таймаут запросов
        fallback_to_simple: Использовать Simple при ошибках Infinity
        
    Returns:
        Инициализированный EmbeddingService
    """
    
    if embedding_type == "infinity":
        logger.info("Creating InfinityEmbeddingService...")
        
        try:
            from .infinity_embedding_service import InfinityEmbeddingService
            
            if not infinity_url:
                logger.warning("Infinity URL not provided, falling back to SimpleEmbedding")
                return SimpleEmbeddingService(dimension=dimension)
            
            service = InfinityEmbeddingService(
                base_url=infinity_url,
                api_key=infinity_api_key,
                dimension=dimension,
                model_name=infinity_model,
                timeout=infinity_timeout,
                fallback_to_simple=fallback_to_simple
            )
            
            # Проверка доступности
            if service.is_available():
                logger.info("InfinityEmbeddingService initialized and available")
                return service
            else:
                logger.warning("Infinity server not available")
                
                if fallback_to_simple:
                    logger.info("Falling back to SimpleEmbeddingService")
                    return SimpleEmbeddingService(dimension=dimension)
                else:
                    return service
        
        except ImportError as e:
            logger.error(f"Cannot import InfinityEmbeddingService: {e}")
            logger.info("Falling back to SimpleEmbeddingService")
            return SimpleEmbeddingService(dimension=dimension)
        
        except Exception as e:
            logger.error(f"Error creating InfinityEmbeddingService: {e}")
            logger.info("Falling back to SimpleEmbeddingService")
            return SimpleEmbeddingService(dimension=dimension)
    
    elif embedding_type == "simple":
        logger.info("Creating SimpleEmbeddingService...")
        return SimpleEmbeddingService(dimension=dimension)
    
    else:
        logger.warning(f"Unknown embedding type: {embedding_type}, using Simple")
        return SimpleEmbeddingService(dimension=dimension)


def create_embedding_service_from_env() -> EmbeddingService:
    """
    Создать сервис эмбеддингов из переменных окружения.
    
    Читает конфигурацию из environment variables:
    - EMBEDDING_TYPE: "simple" или "infinity"
    - EMBEDDING_DIMENSION: размерность
    - INFINITY_BASE_URL: URL сервера
    - INFINITY_API_KEY: API ключ
    - INFINITY_MODEL: модель
    
    Returns:
        Инициализированный EmbeddingService
    """
    import os
    
    return create_embedding_service(
        embedding_type=os.getenv("EMBEDDING_TYPE", "simple"),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
        infinity_url=os.getenv("INFINITY_BASE_URL"),
        infinity_api_key=os.getenv("INFINITY_API_KEY"),
        infinity_model=os.getenv("INFINITY_MODEL", "BAAI/bge-small-en-v1.5"),
        infinity_timeout=int(os.getenv("INFINITY_TIMEOUT", "10")),
        fallback_to_simple=True
    )
