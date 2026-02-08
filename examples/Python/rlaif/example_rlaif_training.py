"""
Пример RLAIF обучения.

Демонстрирует:
- Полный цикл RLAIF (выполнение → оценка → обучение)
- Автоматическое улучшение инструментов
- Статистика обучения
"""

import sys
from pathlib import Path
import os

# Добавляем GraphArchitect
grapharchitect_path = Path(__file__).parent.parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(grapharchitect_path))

print("=" * 70)
print("ПРИМЕР: RLAIF Обучение")
print("=" * 70)
print()

# Проверка API ключа
if not os.getenv("OPENROUTER_API_KEY"):
    print("[WARNING] OPENROUTER_API_KEY не установлен")
    print("Пример покажет структуру без реальных вызовов")
    print()
    print("Для реальной работы:")
    print("  set OPENROUTER_API_KEY=your-key")
    print()
    USE_REAL_LLM = False
else:
    print("[OK] OPENROUTER_API_KEY найден")
    print()
    USE_REAL_LLM = True

# Импорты
from grapharchitect.services.rlaif.llm_critic import LLMCritic
from grapharchitect.services.rlaif.rlaif_trainer import RLAIFTrainer
from grapharchitect.services.training.training_orchestrator import TrainingOrchestrator
from grapharchitect.services.execution.execution_orchestrator import ExecutionOrchestrator
from grapharchitect.services.execution.execution_context import ExecutionStatus
from grapharchitect.services.selection.instrument_selector import InstrumentSelector
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.entities.task_definition import TaskDefinition
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.entities.base_tool import BaseTool


# Создаем простой инструмент для демонстрации
class MockClassifier(BaseTool):
    """Простой классификатор для демонстрации."""
    
    def __init__(self, name, reputation=0.70):
        super().__init__()
        self.metadata.tool_name = name
        self.metadata.reputation = reputation
        self.metadata.training_sample_size = 10
        self.metadata.variance_estimate = 0.1
        self.metadata.mean_cost = 0.02
        self.metadata.mean_time_answer = 2.0
        
        self.input = Connector("text", "question")
        self.output = Connector("text", "category")
    
    def execute(self, input_data):
        return f"[{self.metadata.tool_name}] Классификация: POSITIVE"


print("Шаг 1: Инициализация компонентов")
print("-" * 70)

# Embedding service
embedding_service = SimpleEmbeddingService(dimension=384)

# Training orchestrator
training = TrainingOrchestrator(learning_rate=0.01)

# Execution orchestrator
selector = InstrumentSelector(temperature_constant=1.0)
strategy_finder = GraphStrategyFinder()
orchestrator = ExecutionOrchestrator(
    embedding_service,
    selector,
    strategy_finder
)

print("  [OK] GraphArchitect компоненты инициализированы")

# LLM Critic (если доступен)
if USE_REAL_LLM:
    critic = LLMCritic(
        backend="openrouter",
        model_name="openai/gpt-3.5-turbo",  # Или gpt-4 для лучшей оценки
        temperature=0.2,
        detailed_evaluation=True
    )
    
    print(f"  [OK] LLM Критик: OpenRouter GPT-3.5")
    
    # RLAIF Trainer
    rlaif_trainer = RLAIFTrainer(
        llm_critic=critic,
        training_orchestrator=training,
        min_score_threshold=0.3
    )
    
    print(f"  [OK] RLAIF Trainer инициализирован")
else:
    print("  [SKIP] LLM Критик (нет API ключа)")
    critic = None
    rlaif_trainer = None

print()

# Создаем инструменты
print("Шаг 2: Создание инструментов")
print("-" * 70)

tools = [
    MockClassifier("Classifier-A", reputation=0.70),
    MockClassifier("Classifier-B", reputation=0.65),
]

# Генерируем эмбеддинги
for tool in tools:
    tool.metadata.capabilities_embedding = embedding_service.embed_tool_capabilities(tool)

print(f"  Создано инструментов: {len(tools)}")
for tool in tools:
    print(f"    {tool.metadata.tool_name}: reputation={tool.metadata.reputation:.2f}")

print()

# Выполнение задачи
print("Шаг 3: Выполнение задачи")
print("-" * 70)

task = TaskDefinition(
    description="Классифицировать отзыв клиента",
    input_connector=Connector("text", "question"),
    output_connector=Connector("text", "category"),
    input_data="Отличный продукт, очень доволен покупкой!"
)

context = orchestrator.execute_task(
    task=task,
    available_tools=tools,
    path_limit=1,
    top_k=2
)

print(f"  Статус: {context.status.value}")
print(f"  Результат: {context.result}")
print(f"  Время: {context.total_time:.3f}s")
print()

# RLAIF оценка и обучение
if USE_REAL_LLM and rlaif_trainer:
    print("Шаг 4: RLAIF Оценка и обучение")
    print("-" * 70)
    
    # Оценка и обучение
    train_result = rlaif_trainer.evaluate_and_train(
        context=context,
        task_description=task.description,
        result=context.result
    )
    
    print(f"  Оценка LLM критика: {train_result.average_score:.2f}")
    print(f"  Инструментов обучено: {train_result.tools_updated}")
    print()
    
    print("  Изменения репутации:")
    for tool_name, delta in train_result.improvements.items():
        print(f"    {tool_name}: {delta:+.4f}")
    
    print()
    
    # Статистика
    print("Шаг 5: Статистика оценок")
    print("-" * 70)
    
    stats = rlaif_trainer.get_evaluation_statistics()
    
    print(f"  Всего оценок: {stats['total_evaluations']}")
    print(f"  Средний балл: {stats['average_overall_score']:.2f}")
    print(f"  Правильность: {stats['average_correctness']:.2f}")
    print(f"  Полнота:      {stats['average_completeness']:.2f}")
    
    print()

else:
    print("Шаг 4: RLAIF (демонстрация структуры)")
    print("-" * 70)
    
    print("  БЕЗ OPENROUTER_API_KEY показываем структуру:")
    print()
    print("  1. LLMCritic.evaluate_answer(task, answer)")
    print("     ↓")
    print("  2. LLM оценивает по критериям:")
    print("     - Правильность (0-1)")
    print("     - Полнота (0-1)")
    print("     - Релевантность (0-1)")
    print("     - Ясность (0-1)")
    print("     ↓")
    print("  3. RLAIFTrainer создает FeedbackData")
    print("     ↓")
    print("  4. TrainingOrchestrator обучает инструменты")
    print("     ↓")
    print("  5. Репутация обновляется (Policy Gradient)")
    print()
    
    print("  Пример оценки:")
    print("    overall_score: 0.87")
    print("    correctness: 0.90")
    print("    completeness: 0.85")
    print("    reasoning: 'Ответ правильный и полный...'")
    print()

# Итоги
print("=" * 70)
print("ИТОГИ")
print("=" * 70)
print()
print("RLAIF реализован:")
print("  - LLMCritic: Судья на основе LLM")
print("  - Поддержка OpenRouter и VLLM")
print("  - Детальная оценка по 4 критериям")
print("  - Текстовое обоснование")
print()
print("Применение:")
print("  - Автоматическая оценка качества (без человека)")
print("  - Обучение через Policy Gradient")
print("  - Масштабируемое обучение")
print("  - Consistency оценок")
print()
print("Преимущества vs человек-оценщик:")
print("  - Скорость: мгновенно")
print("  - Стоимость: $0.001-0.01 за оценку")
print("  - Масштабируемость: тысячи оценок")
print("  - Consistency: одинаковые критерии")
print()
print("Для использования:")
print("  1. set OPENROUTER_API_KEY=your-key")
print("  2. python example_llm_critic.py")
print()
print("=" * 70)
