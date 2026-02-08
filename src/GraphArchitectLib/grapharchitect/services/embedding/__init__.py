"""Модуль векторизации текста"""

from .embedding_service import EmbeddingService
from .simple_embedding_service import SimpleEmbeddingService

# Новые сервисы (опциональные)
try:
    from .infinity_embedding_service import InfinityEmbeddingService
    INFINITY_AVAILABLE = True
except ImportError:
    INFINITY_AVAILABLE = False

try:
    from .embedding_factory import create_embedding_service, create_embedding_service_from_env
    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False

# Экспорт
__all__ = ['EmbeddingService', 'SimpleEmbeddingService']

if INFINITY_AVAILABLE:
    __all__.append('InfinityEmbeddingService')

if FACTORY_AVAILABLE:
    __all__.extend(['create_embedding_service', 'create_embedding_service_from_env'])
