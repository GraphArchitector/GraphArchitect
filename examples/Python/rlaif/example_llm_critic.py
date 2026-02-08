"""
Пример использования LLM критика для оценки ответов.

Демонстрирует:
- Создание LLM критика (OpenRouter или VLLM)
- Оценку качества ответов
- Детальную и простую оценку
- RLAIF обучение
"""

import sys
from pathlib import Path
import os

# Добавляем GraphArchitect
grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

print("=" * 70)
print("ПРИМЕР: LLM Критик для RLAIF")
print("=" * 70)
print()

# Проверка API ключа
has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))

if has_openrouter:
    print("[OK] OPENROUTER_API_KEY найден - будет использован OpenRouter")
    backend = "openrouter"
else:
    print("[WARNING] OPENROUTER_API_KEY не установлен")
    print("Пример покажет структуру, но не выполнит реальные вызовы")
    print()
    backend = None

print()

# Импорты
from grapharchitect.services.rlaif.llm_critic import LLMCritic, LLMCriticScore

# Создание критика
if backend == "openrouter":
    print("Шаг 1: Инициализация LLM Критика (OpenRouter)")
    print("-" * 70)
    
    critic = LLMCritic(
        backend="openrouter",
        model_name="openai/gpt-4",  # Можно использовать gpt-3.5-turbo для экономии
        temperature=0.2,  # Низкая для consistency
        detailed_evaluation=True
    )
    
    print(f"  [OK] Критик инициализирован: OpenRouter GPT-4")
    print()
    
    # Тест 1: Хороший ответ
    print("Тест 1: Оценка ХОРОШЕГО ответа")
    print("-" * 70)
    
    task = "Классифицировать отзыв клиента по тональности"
    answer = """[Классификация] Отзыв: ПОЗИТИВНЫЙ

Обоснование:
- Фразы "отличный продукт", "рекомендую" указывают на позитивную тональность
- Нет негативных формулировок
- Общий тон восторженный

Категория: positive
Уверенность: 95%"""
    
    print(f"Задача: {task}")
    print(f"Ответ: {answer[:100]}...")
    print()
    
    score = critic.evaluate_answer(task, answer)
    
    if score:
        print(f"  Общая оценка: {score.overall_score:.2f} (из 1.0)")
        print(f"  Правильность: {score.correctness:.2f}")
        print(f"  Полнота:      {score.completeness:.2f}")
        print(f"  Релевантность: {score.relevance:.2f}")
        print(f"  Ясность:      {score.clarity:.2f}")
        print()
        print(f"  Обоснование: {score.reasoning[:200]}...")
        if score.suggestions:
            print(f"  Предложения: {score.suggestions[:200]}...")
    
    print()
    
    # Тест 2: Плохой ответ
    print("Тест 2: Оценка ПЛОХОГО ответа")
    print("-" * 70)
    
    task = "Ответить на вопрос: Что такое машинное обучение?"
    answer = "Это когда компьютер учится."  # Слишком короткий, неполный
    
    print(f"Задача: {task}")
    print(f"Ответ: {answer}")
    print()
    
    score = critic.evaluate_answer(task, answer)
    
    if score:
        print(f"  Общая оценка: {score.overall_score:.2f} (из 1.0)")
        print(f"  Правильность: {score.correctness:.2f}")
        print(f"  Полнота:      {score.completeness:.2f}")
        print(f"  Релевантность: {score.relevance:.2f}")
        print(f"  Ясность:      {score.clarity:.2f}")
        print()
        print(f"  Обоснование: {score.reasoning[:200]}...")
        if score.suggestions:
            print(f"  Предложения: {score.suggestions[:200]}...")
    
    print()

else:
    # Демонстрация без реального LLM
    print("ДЕМОНСТРАЦИЯ СТРУКТУРЫ (без реальных вызовов)")
    print("=" * 70)
    print()
    
    print("Для реальной работы:")
    print("  1. Установите API ключ:")
    print("     set OPENROUTER_API_KEY=your-key")
    print()
    print("  2. Или используйте VLLM:")
    print("     backend='vllm', vllm_host='http://localhost:8000'")
    print()
    
    # Создание mock критика
    print("Структура LLMCriticScore:")
    print("-" * 70)
    
    mock_score = LLMCriticScore(
        overall_score=0.87,
        correctness=0.90,
        completeness=0.85,
        relevance=0.92,
        clarity=0.80,
        reasoning="Ответ правильный и полный, хорошо структурирован",
        suggestions="Можно добавить примеры для ясности",
        model_used="openai/gpt-4"
    )
    
    print(f"  overall_score: {mock_score.overall_score}")
    print(f"  correctness: {mock_score.correctness}")
    print(f"  completeness: {mock_score.completeness}")
    print(f"  relevance: {mock_score.relevance}")
    print(f"  clarity: {mock_score.clarity}")
    print(f"  reasoning: {mock_score.reasoning}")
    print(f"  suggestions: {mock_score.suggestions}")
    print()

# Итоги
print("=" * 70)
print("ИТОГИ")
print("=" * 70)
print()
print("LLM Критик реализован с поддержкой:")
print("  - OpenRouter (доступ к GPT-4, Claude, Gemini)")
print("  - VLLM (локальные модели)")
print()
print("Оценка включает:")
print("  - Общий балл (0-1)")
print("  - Детальные критерии (correctness, completeness, relevance, clarity)")
print("  - Текстовое обоснование")
print("  - Предложения по улучшению")
print()
print("Применение в RLAIF:")
print("  - Автоматическая оценка качества")
print("  - Обучение через Policy Gradient")
print("  - Без участия человека")
print("  - Consistency и scalability")
print()
print("Для использования:")
print("  1. Установите OPENROUTER_API_KEY или запустите VLLM сервер")
print("  2. Создайте LLMCritic")
print("  3. Используйте в RLAIFTrainer")
print()
print("=" * 70)
