"""
Конфигурация RLAIF для Web API.

Настройки для автоматической оценки качества через LLM критика.
"""

import os

# RLAIF включен/выключен
RLAIF_ENABLED = os.getenv("RLAIF_ENABLED", "false").lower() == "true"

# Бэкенд для LLM критика
RLAIF_BACKEND = os.getenv("RLAIF_BACKEND", "openrouter")  # openrouter или vllm

# OpenRouter настройки
RLAIF_OPENROUTER_MODEL = os.getenv("RLAIF_OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
RLAIF_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Или отдельный RLAIF_API_KEY

# VLLM настройки
RLAIF_VLLM_HOST = os.getenv("RLAIF_VLLM_HOST", "http://localhost:8000")
RLAIF_VLLM_MODEL = os.getenv("RLAIF_VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Параметры оценки
RLAIF_TEMPERATURE = float(os.getenv("RLAIF_TEMPERATURE", "0.2"))
RLAIF_DETAILED = os.getenv("RLAIF_DETAILED", "true").lower() == "true"

# Пороги
RLAIF_MIN_SCORE_THRESHOLD = float(os.getenv("RLAIF_MIN_SCORE", "0.3"))

# Автообучение
RLAIF_AUTO_TRAIN = os.getenv("RLAIF_AUTO_TRAIN", "true").lower() == "true"
