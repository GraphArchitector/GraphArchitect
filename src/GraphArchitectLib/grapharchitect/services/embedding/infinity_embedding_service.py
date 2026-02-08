"""
Сервис эмбеддингов на основе Infinity API.

Использует внешний Infinity сервер для получения качественных
семантических эмбеддингов вместо простого хеширования.
"""

from typing import List, Optional
import logging

from .embedding_service import EmbeddingService

# Импорт InfinityEmbedder
try:
    from ...tools.ApiTools.InfinityTool.Embedder import InfinityEmbedder
    INFINITY_AVAILABLE = True
except ImportError:
    INFINITY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Импорт для TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...entities.base_tool import BaseTool


class InfinityEmbeddingService(EmbeddingService):
    """
    Сервис эмбеддингов через Infinity API.
    
    Преимущества:
    - Высокое качество семантических эмбеддингов
    - Поддержка различных моделей (E5, BGE, etc.)
    - Масштабируемость через внешний сервис
    - Batch processing для эффективности
    
    Требования:
    - Запущенный Infinity сервер
    - Сетевое соединение
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        dimension: int = 384,
        model_name: str = "BAAI/bge-small-en-v1.5",
        timeout: int = 10,
        fallback_to_simple: bool = True
    ):
        """
        Инициализация сервиса.
        
        Args:
            base_url: URL Infinity сервера (например, "http://localhost:7997")
            api_key: API ключ (если требуется)
            dimension: Размерность эмбеддингов (зависит от модели)
            model_name: Название модели эмбеддингов
            timeout: Таймаут запросов в секундах
            fallback_to_simple: Использовать SimpleEmbedding при ошибках
        """
        if not INFINITY_AVAILABLE:
            raise ImportError(
                "InfinityEmbedder not available. "
                "Check that InfinityTool is in the correct path."
            )
        
        self._infinity = InfinityEmbedder(base_url=base_url, api_key=api_key)
        self._dimension = dimension
        self._model_name = model_name
        self._timeout = timeout
        self._fallback_to_simple = fallback_to_simple
        
        # Fallback на SimpleEmbedding если включен
        self._fallback_service = None
        if fallback_to_simple:
            from .simple_embedding_service import SimpleEmbeddingService
            self._fallback_service = SimpleEmbeddingService(dimension=dimension)
        
        logger.info(f"InfinityEmbeddingService initialized (server: {base_url}, model: {model_name})")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Создать эмбеддинг для текста через Infinity API.
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Вектор эмбеддинга
        """
        if not text:
            return [0.0] * self._dimension
        
        try:
            # Вызов Infinity API
            result = self._infinity.get_embedding(text, timeout=self._timeout)
            
            # Проверка на ошибку
            if "error" in result:
                logger.error(f"Infinity API error: {result['error']}")
                
                if self._fallback_service:
                    logger.warning("Falling back to SimpleEmbedding")
                    return self._fallback_service.embed_text(text)
                else:
                    return [0.0] * self._dimension
            
            # Извлечение вектора из ответа
            # Формат зависит от Infinity API, может быть:
            # {"embedding": [...]} или {"data": [{"embedding": [...]}]}
            embedding = None
            
            if "embedding" in result:
                embedding = result["embedding"]
            elif "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                embedding = result["data"][0].get("embedding")
            
            if embedding and isinstance(embedding, list):
                # Проверка размерности
                if len(embedding) != self._dimension:
                    logger.warning(
                        f"Embedding dimension mismatch: expected {self._dimension}, got {len(embedding)}"
                    )
                
                return embedding
            else:
                logger.error(f"Unexpected Infinity response format: {result}")
                
                if self._fallback_service:
                    return self._fallback_service.embed_text(text)
                else:
                    return [0.0] * self._dimension
        
        except Exception as e:
            logger.error(f"Error calling Infinity API: {e}")
            
            if self._fallback_service:
                logger.warning("Falling back to SimpleEmbedding")
                return self._fallback_service.embed_text(text)
            else:
                return [0.0] * self._dimension
    
    def embed_tool_capabilities(self, tool: 'BaseTool') -> List[float]:
        """
        Создать эмбеддинг для возможностей инструмента.
        
        Args:
            tool: Инструмент для векторизации
            
        Returns:
            Вектор эмбеддинга
        """
        # Формируем текст из метаданных
        text_parts = []
        
        if tool.metadata.tool_name:
            text_parts.append(tool.metadata.tool_name)
        
        if tool.metadata.description:
            text_parts.append(tool.metadata.description)
        
        # Добавляем форматы коннекторов
        text_parts.append(f"input:{tool.input.format}")
        text_parts.append(f"output:{tool.output.format}")
        
        combined_text = " ".join(text_parts)
        return self.embed_text(combined_text)
    
    def compute_similarity(
        self,
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """
        Вычислить косинусное сходство между векторами.
        
        Args:
            vector1: Первый вектор
            vector2: Второй вектор
            
        Returns:
            Косинусное сходство в диапазоне [0, 1]
        """
        if len(vector1) != len(vector2):
            return 0.0
        
        import math
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = math.sqrt(sum(a * a for a in vector1))
        norm2 = math.sqrt(sum(b * b for b in vector2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Косинусное сходство в диапазоне [-1, 1]
        cos_sim = dot_product / (norm1 * norm2)
        
        # Нормализуем в [0, 1]
        return (cos_sim + 1.0) / 2.0
    
    @property
    def embedding_dimension(self) -> int:
        """Размерность эмбеддингов."""
        return self._dimension
    
    def is_available(self) -> bool:
        """
        Проверить доступность Infinity сервера.
        
        Returns:
            True если сервер отвечает
        """
        try:
            # Пробуем получить эмбеддинг для тестового текста
            result = self._infinity.get_embedding("test", timeout=2)
            return "error" not in result
        except Exception:
            return False
