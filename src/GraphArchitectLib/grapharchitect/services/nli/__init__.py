"""Модуль естественно-языкового интерфейса (ЕЯИ / NLI)"""

from .nli_dataset_item import NLIDatasetItem
from .knn_few_shot_retriever import KNNFewShotRetriever, ScoredExample, DatasetStatistics
from .connector_info_aggregator import ConnectorInfoAggregator
from .natural_language_interface import NaturalLanguageInterface, NLIParseResult

# Новые ретриверы (опциональные)
try:
    from .faiss_knn_retriever import FaissKNNRetriever
    FAISS_RETRIEVER_AVAILABLE = True
except ImportError:
    FAISS_RETRIEVER_AVAILABLE = False

try:
    from .retriever_factory import create_knn_retriever, create_knn_retriever_from_env
    RETRIEVER_FACTORY_AVAILABLE = True
except ImportError:
    RETRIEVER_FACTORY_AVAILABLE = False

# LLM-based NLI (опционально)
try:
    from .llm_nli_service import LLMNLIService
    LLM_NLI_AVAILABLE = True
except ImportError:
    LLM_NLI_AVAILABLE = False

try:
    from .qwen_nli_service import QwenNLIService
    QWEN_NLI_AVAILABLE = True
except ImportError:
    QWEN_NLI_AVAILABLE = False

try:
    from .nli_service_factory import create_nli_service, create_nli_service_from_env
    NLI_FACTORY_AVAILABLE = True
except ImportError:
    NLI_FACTORY_AVAILABLE = False

# Экспорт
__all__ = [
    'NLIDatasetItem',
    'KNNFewShotRetriever',
    'ScoredExample',
    'DatasetStatistics',
    'ConnectorInfoAggregator',
    'NaturalLanguageInterface',
    'NLIParseResult'
]

if FAISS_RETRIEVER_AVAILABLE:
    __all__.append('FaissKNNRetriever')

if RETRIEVER_FACTORY_AVAILABLE:
    __all__.extend(['create_knn_retriever', 'create_knn_retriever_from_env'])

if LLM_NLI_AVAILABLE:
    __all__.append('LLMNLIService')

if QWEN_NLI_AVAILABLE:
    __all__.append('QwenNLIService')

if NLI_FACTORY_AVAILABLE:
    __all__.extend(['create_nli_service', 'create_nli_service_from_env'])
