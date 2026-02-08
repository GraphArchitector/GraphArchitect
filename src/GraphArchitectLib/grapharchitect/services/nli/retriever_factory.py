"""
Фабрика для создания k-NN ретриверов.

Автоматически выбирает между наивным поиском и Faiss
на основе конфигурации и доступности библиотек.
"""

import logging
from typing import Optional

from ..embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def create_knn_retriever(
    embedding_service: EmbeddingService,
    retriever_type: str = "naive",
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    faiss_index_type: str = "FlatIP"
):
    """
    Создать k-NN ретривер на основе конфигурации.
    
    Args:
        embedding_service: Сервис эмбеддингов
        retriever_type: Тип ретривера ("naive", "faiss")
        vector_weight: Вес векторной схожести
        text_weight: Вес текстовой схожести
        faiss_index_type: Тип Faiss индекса (FlatIP, FlatL2, HNSW)
        
    Returns:
        Инициализированный ретривер
    """
    
    if retriever_type == "faiss":
        logger.info("Creating FaissKNNRetriever...")
        
        try:
            from .faiss_knn_retriever import FaissKNNRetriever, FAISS_AVAILABLE
            
            if not FAISS_AVAILABLE:
                logger.warning("Faiss not available, falling back to naive k-NN")
                from .knn_few_shot_retriever import KNNFewShotRetriever
                return KNNFewShotRetriever(
                    embedding_service=embedding_service,
                    vector_weight=vector_weight,
                    text_weight=text_weight
                )
            
            retriever = FaissKNNRetriever(
                embedding_service=embedding_service,
                vector_weight=vector_weight,
                text_weight=text_weight,
                use_faiss=True,
                index_type=faiss_index_type
            )
            
            logger.info(f"FaissKNNRetriever created (index: {faiss_index_type})")
            return retriever
        
        except ImportError as e:
            logger.error(f"Cannot import FaissKNNRetriever: {e}")
            logger.info("Falling back to KNNFewShotRetriever")
            from .knn_few_shot_retriever import KNNFewShotRetriever
            return KNNFewShotRetriever(
                embedding_service=embedding_service,
                vector_weight=vector_weight,
                text_weight=text_weight
            )
    
    elif retriever_type == "naive":
        logger.info("Creating KNNFewShotRetriever (naive)...")
        from .knn_few_shot_retriever import KNNFewShotRetriever
        return KNNFewShotRetriever(
            embedding_service=embedding_service,
            vector_weight=vector_weight,
            text_weight=text_weight
        )
    
    else:
        logger.warning(f"Unknown retriever type: {retriever_type}, using naive")
        from .knn_few_shot_retriever import KNNFewShotRetriever
        return KNNFewShotRetriever(
            embedding_service=embedding_service,
            vector_weight=vector_weight,
            text_weight=text_weight
        )


def create_knn_retriever_from_env(embedding_service: EmbeddingService):
    """
    Создать k-NN ретривер из переменных окружения.
    
    Читает конфигурацию:
    - KNN_TYPE: "naive" или "faiss"
    - FAISS_INDEX_TYPE: "FlatIP", "FlatL2", "HNSW"
    - KNN_VECTOR_WEIGHT: вес векторной схожести
    - KNN_TEXT_WEIGHT: вес текстовой схожести
    
    Args:
        embedding_service: Сервис эмбеддингов
        
    Returns:
        Инициализированный ретривер
    """
    import os
    
    return create_knn_retriever(
        embedding_service=embedding_service,
        retriever_type=os.getenv("KNN_TYPE", "naive"),
        vector_weight=float(os.getenv("KNN_VECTOR_WEIGHT", "0.7")),
        text_weight=float(os.getenv("KNN_TEXT_WEIGHT", "0.3")),
        faiss_index_type=os.getenv("FAISS_INDEX_TYPE", "FlatIP")
    )
