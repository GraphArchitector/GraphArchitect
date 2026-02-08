"""
RLAIF (Reinforcement Learning from AI Feedback).

Модуль для автоматической оценки качества и обучения
с использованием LLM в роли критика/судьи.

Компоненты:
- LLMCritic: Судья на основе LLM (VLLM или OpenRouter)
- RLAIFTrainer: Тренер с автоматической оценкой и обучением
"""

from .llm_critic import LLMCritic, LLMCriticScore
from .rlaif_trainer import RLAIFTrainer, RLAIFTrainingResult

__all__ = [
    "LLMCritic",
    "LLMCriticScore",
    "RLAIFTrainer",
    "RLAIFTrainingResult"
]
