"""Простой сервис векторизации (заглушка)"""

from typing import List
import hashlib
import math

from .embedding_service import EmbeddingService

# Используем TYPE_CHECKING для избежания циклических импортов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...entities.base_tool import BaseTool


class SimpleEmbeddingService(EmbeddingService):
    """
    Простая реализация сервиса векторизации.
    
    ВАЖНО: Это заглушка для демонстрации.
    В продакшене использовать:
    - OpenAI Embeddings API
    - Sentence Transformers
    - Другие модели эмбеддингов
    """
    
    def __init__(self, dimension: int = 384):
        """
        Инициализация сервиса.
        
        Args:
            dimension: Размерность эмбеддингов
        """
        self._dimension = dimension
    
    def embed_text(self, text: str) -> List[float]:
        """
        Создать эмбеддинг для текста.
        
        Простая реализация на основе хеширования.
        В реальности использовать нейронные сети.
        """
        if not text:
            return [0.0] * self._dimension
        
        # Хешируем текст для получения псевдослучайного вектора
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Преобразуем в вектор
        vector = []
        for i in range(self._dimension):
            # Берем байты циклически
            byte_value = text_hash[i % len(text_hash)]
            # Нормализуем в диапазон [-1, 1]
            vector.append((byte_value / 127.5) - 1.0)
        
        # Нормализуем вектор
        return self._normalize(vector)
    
    def embed_tool_capabilities(self, tool: 'BaseTool') -> List[float]:
        """
        Создать эмбеддинг для возможностей инструмента.
        
        Используем описание инструмента и форматы коннекторов.
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
        Вычислить косинусное сходство.
        
        Args:
            vector1: Первый вектор
            vector2: Второй вектор
            
        Returns:
            Косинусное сходство (0-1, нормализовано)
        """
        if len(vector1) != len(vector2):
            return 0.0
        
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
        """Размерность эмбеддингов"""
        return self._dimension
    
    def _normalize(self, vector: List[float]) -> List[float]:
        """Нормализовать вектор"""
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]
