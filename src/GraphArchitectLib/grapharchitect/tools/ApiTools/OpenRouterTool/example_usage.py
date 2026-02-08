"""
Примеры использования OpenRouter инструментов.

Демонстрирует:
- Базовое использование OpenRouterChatTool
- Классификацию с OpenRouterClassifierTool
- Суммаризацию с OpenRouterSummarizerTool
- Анализ с OpenRouterAnalyzerTool
- Сравнение разных моделей
"""

import os
import sys
from pathlib import Path

# Добавляем путь к grapharchitect
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from grapharchitect.tools.ApiTools.OpenRouterTool import (
    OpenRouterChatTool,
    OpenRouterClassifierTool,
    OpenRouterSummarizerTool,
    OpenRouterAnalyzerTool,
    OpenRouterConfig
)


def example_1_basic_chat():
    """Пример 1: Базовый чат"""
    print("\n" + "="*70)
    print("ПРИМЕР 1: Базовый чат с GPT-3.5")
    print("="*70 + "\n")
    
    # Создаем инструмент
    tool = OpenRouterChatTool(
        model_key="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant."
    )
    
    # Задаем вопросы
    questions = [
        "What is artificial intelligence?",
        "Explain neural networks in simple terms.",
        "What are the benefits of machine learning?"
    ]
    
    for q in questions:
        print(f"Q: {q}")
        answer = tool.execute(q)
        print(f"A: {answer}\n")


def example_2_classification():
    """Пример 2: Классификация"""
    print("\n" + "="*70)
    print("ПРИМЕР 2: Классификация отзывов")
    print("="*70 + "\n")
    
    # Создаем классификатор
    classifier = OpenRouterClassifierTool(
        model_key="gpt-3.5-turbo",
        categories=["positive", "negative", "neutral"]
    )
    
    # Тестовые отзывы
    reviews = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality. Complete waste of money.",
        "It's okay. Does what it's supposed to do.",
        "Love it! Highly recommend!",
        "Not worth the price. Very disappointed."
    ]
    
    for review in reviews:
        sentiment = classifier.execute(review)
        print(f"Review: {review[:50]}...")
        print(f"Sentiment: {sentiment}\n")


def example_3_summarization():
    """Пример 3: Суммаризация"""
    print("\n" + "="*70)
    print("ПРИМЕР 3: Суммаризация текста")
    print("="*70 + "\n")
    
    # Создаем суммаризатор
    summarizer = OpenRouterSummarizerTool(
        model_key="gpt-3.5-turbo",
        max_summary_words=50
    )
    
    # Длинный текст
    long_text = """
    Artificial intelligence (AI) is transforming the world in unprecedented ways. 
    From healthcare to finance, from transportation to entertainment, AI technologies 
    are being deployed to solve complex problems and improve human lives. Machine 
    learning algorithms can now diagnose diseases with accuracy rivaling human doctors, 
    autonomous vehicles are becoming a reality, and natural language processing enables 
    computers to understand and generate human language with remarkable fluency. 
    However, these advancements also bring challenges, including ethical concerns 
    about bias in AI systems, questions about job displacement, and the need for 
    robust AI governance frameworks.
    """
    
    summary = summarizer.execute(long_text)
    
    print(f"Оригинал ({len(long_text)} символов):")
    print(long_text[:200] + "...\n")
    print(f"Сводка:")
    print(summary + "\n")


def example_4_model_comparison():
    """Пример 4: Сравнение моделей"""
    print("\n" + "="*70)
    print("ПРИМЕР 4: Сравнение разных моделей")
    print("="*70 + "\n")
    
    # Создаем инструменты с разными моделями
    models = ["gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"]
    tools = {}
    
    for model_key in models:
        try:
            tool = OpenRouterChatTool(model_key=model_key)
            tools[model_key] = tool
        except Exception as e:
            print(f"⚠️ Не удалось создать {model_key}: {e}")
    
    if not tools:
        print("❌ Модели не доступны (проверьте API ключ)")
        return
    
    # Одинаковый вопрос
    question = "Explain the concept of recursion in programming."
    
    print(f"Вопрос: {question}\n")
    
    for model_key, tool in tools.items():
        print(f"{'='*70}")
        print(f"Модель: {tool.metadata.tool_name}")
        print(f"{'='*70}")
        
        answer = tool.execute(question)
        print(f"{answer}\n")


def example_5_list_models():
    """Пример 5: Список доступных моделей"""
    print("\n" + "="*70)
    print("ПРИМЕР 5: Доступные модели")
    print("="*70 + "\n")
    
    # Все модели
    models = OpenRouterConfig.list_models()
    
    print(f"Всего моделей в конфигурации: {len(models)}\n")
    
    # Группируем по провайдеру
    by_provider = {}
    for key, config in models.items():
        if config.provider not in by_provider:
            by_provider[config.provider] = []
        by_provider[config.provider].append((key, config))
    
    for provider, models_list in sorted(by_provider.items()):
        print(f"\n{provider.upper()}:")
        print("-" * 70)
        
        for key, config in models_list:
            print(f"  {key:20} {config.display_name:30} ${config.cost_per_1m_tokens:6.2f}/1M")
    
    # Самая дешевая
    cheapest = OpenRouterConfig.get_cheapest_model()
    print(f"\n💰 Самая дешевая: {cheapest.display_name} (${cheapest.cost_per_1m_tokens}/1M)")
    
    # Лучшая
    best = OpenRouterConfig.get_best_model()
    print(f"🏆 Лучшая: {best.display_name}")


def main():
    """Запуск всех примеров"""
    print("\n" + "="*70)
    print(" ПРИМЕРЫ OPENROUTER ИНТЕГРАЦИИ")
    print("="*70)
    
    # Проверка API ключа
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("\n⚠️ ВНИМАНИЕ: OPENROUTER_API_KEY не установлен")
        print("\nДля выполнения примеров:")
        print("  1. Получите API ключ на https://openrouter.ai/keys")
        print("  2. Установите переменную окружения:")
        print("     export OPENROUTER_API_KEY=sk-or-v1-...")
        print("\nПримеры будут показаны в демо-режиме (без реальных запросов)\n")
        
        # Показываем только пример со списком моделей
        example_5_list_models()
        return
    
    print(f"\n✅ API ключ найден: {api_key[:20]}...")
    
    try:
        # Запускаем примеры
        example_1_basic_chat()
        example_2_classification()
        example_3_summarization()
        example_4_model_comparison()
        example_5_list_models()
        
        print("\n" + "="*70)
        print(" ✓ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ")
        print("="*70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
