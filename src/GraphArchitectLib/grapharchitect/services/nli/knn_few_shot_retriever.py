"""k-NN Few-Shot ретривер для поиска похожих примеров задач"""

from typing import List, Set
import math
from dataclasses import dataclass

from ...entities.connectors.available_connector_info import AvailableConnectorInfo
from ..embedding.embedding_service import EmbeddingService
from .nli_dataset_item import NLIDatasetItem


@dataclass
class ScoredExample:
    """Пример с оценкой схожести"""
    
    item: NLIDatasetItem
    vector_score: float  # Векторная схожесть
    text_score: float    # Текстовая схожесть
    final_score: float   # Финальная оценка


@dataclass
class DatasetStatistics:
    """Статистика датасета"""
    
    total_examples: int = 0
    unique_file_types: int = 0
    unique_complex_types: int = 0
    unique_knowledge_domains: int = 0
    examples_with_embeddings: int = 0


class KNNFewShotRetriever:
    """
    k-NN Few-Shot ретривер для поиска похожих примеров задач.
    
    Использует комбинированный поиск:
    - Векторный (косинусное сходство эмбеддингов)
    - Полнотекстовый (Jaccard similarity)
    - Фильтрацию по доступным типам
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ):
        """
        Инициализация ретривера.
        
        Args:
            embedding_service: Сервис векторизации
            vector_weight: Вес векторной схожести
            text_weight: Вес текстовой схожести
        """
        self._dataset: List[NLIDatasetItem] = []
        self._embedding_service = embedding_service
        self._vector_weight = vector_weight
        self._text_weight = text_weight
    
    def add_example(self, item: NLIDatasetItem):
        """
        Добавить пример в датасет.
        
        Args:
            item: Элемент датасета
        """
        # Создать эмбеддинг если его нет
        if item.task_embedding is None and item.task_text:
            item.task_embedding = self._embedding_service.embed_text(
                item.task_text
            )
        
        self._dataset.append(item)
    
    def load_dataset(self, items: List[NLIDatasetItem]):
        """
        Загрузить датасет из списка.
        
        Args:
            items: Список элементов
        """
        for item in items:
            self.add_example(item)
    
    def find_similar_examples(
        self,
        task_text: str,
        available_connectors: AvailableConnectorInfo,
        k: int = 3,
        min_score: float = 0.0
    ) -> List[ScoredExample]:
        """
        Найти k ближайших примеров с учетом доступных коннекторов.
        
        Args:
            task_text: Текст задачи
            available_connectors: Доступные коннекторы
            k: Количество примеров
            min_score: Минимальный порог схожести
            
        Returns:
            Список похожих примеров
        """
        if not task_text:
            raise ValueError("Task text cannot be empty")
        
        # 1. Создать эмбеддинг запроса
        query_embedding = self._embedding_service.embed_text(task_text)
        
        # 2. Предварительная фильтрация по доступным типам
        filtered_dataset = self._filter_by_available_types(
            self._dataset,
            available_connectors
        )
        
        if not filtered_dataset:
            # Если после фильтрации ничего не нашли, используем весь датасет
            filtered_dataset = self._dataset
        
        # 3. Вычислить комбинированный score для каждого примера
        scored_examples = []
        
        for item in filtered_dataset:
            # Векторная схожесть
            vector_score = 0.0
            if item.task_embedding and query_embedding:
                vector_score = self._cosine_similarity(
                    query_embedding,
                    item.task_embedding
                )
            
            # Текстовая схожесть
            text_score = self._compute_text_similarity(
                task_text,
                item.task_text
            )
            
            # Комбинированный score
            final_score = (
                self._vector_weight * vector_score +
                self._text_weight * text_score
            )
            
            if final_score >= min_score:
                scored_examples.append(ScoredExample(
                    item=item,
                    vector_score=vector_score,
                    text_score=text_score,
                    final_score=final_score
                ))
        
        # 4. Отсортировать и взять топ-k
        scored_examples.sort(key=lambda x: x.final_score, reverse=True)
        return scored_examples[:k]
    
    def _filter_by_available_types(
        self,
        dataset: List[NLIDatasetItem],
        available: AvailableConnectorInfo
    ) -> List[NLIDatasetItem]:
        """Фильтрация по доступным типам"""
        return [
            item for item in dataset
            if self._is_item_compatible(item, available)
        ]
    
    def _is_item_compatible(
        self,
        item: NLIDatasetItem,
        available: AvailableConnectorInfo
    ) -> bool:
        """Проверить совместимость элемента с доступными типами"""
        # Проверяем file types
        if item.file_types:
            if not any(ft in available.file_types for ft in item.file_types):
                return False
        
        # Проверяем complex types
        if item.complex_types:
            if not any(ct in available.complex_types for ct in item.complex_types):
                return False
        
        # Knowledge domain НЕ блокирует поиск
        return True
    
    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float]
    ) -> float:
        """Косинусное сходство между векторами"""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Полнотекстовая схожесть на основе общих слов (Jaccard).
        
        В продакшене использовать BM25 или TF-IDF.
        """
        if not text1 or not text2:
            return 0.0
        
        tokens1 = self._tokenize(text1.lower())
        tokens2 = self._tokenize(text2.lower())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _tokenize(self, text: str) -> Set[str]:
        """Простая токенизация текста"""
        separators = [' ', ',', '.', '!', '?', ';', ':', '-', '\n', '\r', '\t']
        
        for sep in separators:
            text = text.replace(sep, ' ')
        
        tokens = {
            token.strip()
            for token in text.split()
            if len(token.strip()) > 2  # Игнорируем короткие слова
        }
        
        return tokens
    
    def get_statistics(self) -> DatasetStatistics:
        """Получить статистику датасета"""
        all_file_types = set()
        all_complex_types = set()
        all_knowledge_domains = set()
        
        for item in self._dataset:
            all_file_types.update(item.file_types)
            all_complex_types.update(item.complex_types)
            all_knowledge_domains.update(item.knowledge_domains)
        
        return DatasetStatistics(
            total_examples=len(self._dataset),
            unique_file_types=len(all_file_types),
            unique_complex_types=len(all_complex_types),
            unique_knowledge_domains=len(all_knowledge_domains),
            examples_with_embeddings=sum(
                1 for item in self._dataset if item.task_embedding
            )
        )
