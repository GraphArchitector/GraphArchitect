"""Естественно-языковой интерфейс (ЕЯИ) для преобразования текста задачи в коннекторы"""

from typing import List, Optional, TYPE_CHECKING
import json
from dataclasses import dataclass

from ...entities.base_tool import BaseTool
from ...entities.connectors.task_representation import TaskRepresentation
from ...entities.connectors.available_connector_info import AvailableConnectorInfo
from ..embedding.embedding_service import EmbeddingService
from .knn_few_shot_retriever import KNNFewShotRetriever, ScoredExample
from .connector_info_aggregator import ConnectorInfoAggregator
from .nli_dataset_item import NLIDatasetItem

if TYPE_CHECKING:
    from typing import Union
    # Для поддержки разных типов ретриверов
    RetrieverType = Union[KNNFewShotRetriever, 'FaissKNNRetriever']


@dataclass
class NLIParseResult:
    """Результат парсинга задачи ЕЯИ"""
    
    # Успешность парсинга
    success: bool = False
    
    # Предсказанное представление задачи (коннекторы)
    task_representation: Optional[TaskRepresentation] = None
    
    # Похожие примеры, использованные для предсказания
    similar_examples: Optional[List[ScoredExample]] = None
    
    # Информация о доступных коннекторах
    available_connectors: Optional[AvailableConnectorInfo] = None
    
    # Уверенность в предсказании (0-1)
    confidence: float = 0.0
    
    # Сообщение об ошибке (если success = False)
    error_message: str = ""


class NaturalLanguageInterface:
    """
    Естественно-языковой интерфейс (ЕЯИ) для преобразования текста задачи в коннекторы.
    
    Реализует пункт 1 из описания системы:
    Задача на естественном языке -> {входной коннектор, выходной коннектор, описание}
    
    Использует k-NN few-shot для поиска похожих примеров.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        retriever: Optional['KNNFewShotRetriever'] = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ):
        """
        Инициализация ЕЯИ.
        
        Args:
            embedding_service: Сервис векторизации
            retriever: K-NN ретривер (если None, создается автоматически)
            vector_weight: Вес векторной схожести (используется если retriever=None)
            text_weight: Вес текстовой схожести (используется если retriever=None)
        """
        # Используем переданный retriever или создаем новый
        if retriever is not None:
            self._retriever = retriever
        else:
            self._retriever = KNNFewShotRetriever(
                embedding_service,
                vector_weight,
                text_weight
            )
        
        self._aggregator = ConnectorInfoAggregator()
        self._embedding_service = embedding_service
    
    def load_dataset(self, examples: List[NLIDatasetItem]):
        """
        Загрузить датасет примеров для few-shot.
        
        Args:
            examples: Список примеров
        """
        self._retriever.load_dataset(examples)
    
    def add_example(self, example: NLIDatasetItem):
        """
        Добавить новый пример в датасет.
        
        Args:
            example: Пример для добавления
        """
        self._retriever.add_example(example)
    
    def parse_task(
        self,
        task_text: str,
        available_tools: List[BaseTool],
        k: int = 3
    ) -> NLIParseResult:
        """
        Преобразовать текст задачи в коннекторы с использованием k-NN few-shot.
        
        Args:
            task_text: Текст задачи на естественном языке
            available_tools: Доступные инструменты
            k: Количество примеров для few-shot
            
        Returns:
            Результат парсинга с коннекторами
        """
        if not task_text:
            return NLIParseResult(
                success=False,
                error_message="Task text cannot be empty"
            )
        
        # 1. Агрегировать информацию о доступных коннекторах
        available_connectors = self._aggregator.aggregate_from_tools(
            available_tools
        )
        
        # 2. Найти k похожих примеров
        similar_examples = self._retriever.find_similar_examples(
            task_text,
            available_connectors,
            k,
            min_score=0.1
        )
        
        if not similar_examples:
            return NLIParseResult(
                success=False,
                error_message=(
                    "Не найдено похожих примеров. "
                    "Возможно, отсутствуют необходимые типы данных или коннекторов."
                )
            )
        
        # 3. Предсказать коннекторы на основе примеров
        prediction = self._predict_connectors(
            task_text,
            similar_examples,
            available_connectors
        )
        
        return NLIParseResult(
            success=True,
            task_representation=prediction,
            similar_examples=similar_examples,
            available_connectors=available_connectors,
            confidence=similar_examples[0].final_score
        )
    
    def _predict_connectors(
        self,
        task_text: str,
        examples: List[ScoredExample],
        available_connectors: AvailableConnectorInfo
    ) -> TaskRepresentation:
        """
        Предсказать коннекторы на основе похожих примеров.
        
        Простая эвристика: используем самый похожий пример.
        В продакшене здесь должна быть ML-модель или LLM.
        
        Возможные стратегии:
        1. Взвешенное голосование по топ-k примерам
        2. Использовать только самый похожий
        3. LLM с промптом, содержащим примеры
        """
        # Берём самый похожий пример
        best_example = examples[0]
        
        # Клонируем representation из лучшего примера
        if best_example.item.representation:
            return self._clone_representation(
                best_example.item.representation
            )
        
        # Если representation нет, возвращаем пустой
        return TaskRepresentation()
    
    def _clone_representation(
        self,
        source: TaskRepresentation
    ) -> TaskRepresentation:
        """Клонировать TaskRepresentation"""
        if source is None:
            return TaskRepresentation()
        
        # Простое клонирование через JSON
        json_str = json.dumps(source, default=lambda o: o.__dict__)
        json_obj = json.loads(json_str)
        
        # TODO: Реализовать правильное десериализацию
        return TaskRepresentation(
            input_connector=source.input_connector,
            output_connector=source.output_connector
        )
    
    def get_dataset_statistics(self) -> 'DatasetStatistics':
        """Получить статистику датасета"""
        from .knn_few_shot_retriever import DatasetStatistics
        return self._retriever.get_statistics()
    
    def explain_prediction(self, result: NLIParseResult) -> str:
        """
        Объяснить предсказание (для отладки и интерпретируемости).
        
        Args:
            result: Результат парсинга
            
        Returns:
            Текстовое объяснение
        """
        if not result.success or not result.similar_examples:
            return "Предсказание не удалось"
        
        lines = []
        lines.append("=== Объяснение предсказания ===")
        lines.append(f"Confidence: {result.confidence:.3f}")
        lines.append(f"\nНайдено похожих примеров: {len(result.similar_examples)}")
        
        for example in result.similar_examples:
            lines.append(f"\n--- Пример (score: {example.final_score:.3f}) ---")
            lines.append(f"Векторная схожесть: {example.vector_score:.3f}")
            lines.append(f"Текстовая схожесть: {example.text_score:.3f}")
            lines.append(f"Текст: {example.item.task_text}")
            
            if example.item.representation:
                rep = example.item.representation
                if rep.input_connector:
                    lines.append(
                        f"Input: {rep.input_connector.data_type} | "
                        f"{rep.input_connector.semantic_type}"
                    )
                if rep.output_connector:
                    lines.append(
                        f"Output: {rep.output_connector.data_type} | "
                        f"{rep.output_connector.semantic_type}"
                    )
        
        lines.append("\n=== Предсказанные коннекторы ===")
        if result.task_representation:
            lines.append(
                json.dumps(
                    result.task_representation,
                    default=lambda o: o.__dict__,
                    indent=2,
                    ensure_ascii=False
                )
            )
        
        return "\n".join(lines)
