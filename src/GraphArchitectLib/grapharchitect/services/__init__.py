"""Модуль сервисов - основная логика работы системы"""

from . import nli
from . import embedding
from . import execution
from . import selection
from . import feedback
from . import training

__all__ = ['nli', 'embedding', 'execution', 'selection', 'feedback', 'training']
