"""
Тестовое приложение для демонстрации работы AI-оркестратора
с возможностью выбора алгоритма поиска путей

Портировано с C# на Python
Обновлено для GraphArchitect 3.0
"""

import sys
import os
from pathlib import Path

# Добавляем путь к библиотеке grapharchitect
project_root = Path(__file__).parent.parent.parent / "src" / "GraphArchitectLib"
sys.path.insert(0, str(project_root))

from typing import List
from datetime import datetime

from grapharchitect.entities.base_tool import BaseTool
from grapharchitect.entities.task_definition import TaskDefinition
from grapharchitect.entities.connectors.connector import Connector
from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
from grapharchitect.services.selection.instrument_selector import InstrumentSelector
from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
from grapharchitect.services.execution.execution_orchestrator import ExecutionOrchestrator
from grapharchitect.services.execution.execution_context import ExecutionContext
from grapharchitect.services.feedback.simple_critic import SimpleCritic
from grapharchitect.services.feedback.feedback_collector import FeedbackCollector
from grapharchitect.services.feedback.feedback_data import FeedbackData, FeedbackSource
from grapharchitect.services.training.training_orchestrator import TrainingOrchestrator
from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm

from reactive_tool import ReactiveTool


def main():
    """Главная функция"""
    print("\n" + "=" * 70)
    print("            Тестирование AI-оркестратора")
    print("=" * 70 + "\n")
    
    run_test()
    
    print("\nНажмите Enter для выхода...")
    input()


def run_test():
    """Выполнить тестовый прогон"""
    
    # Инициализация компонентов системы
    print("Инициализация компонентов...")
    embedding_service = SimpleEmbeddingService(dimension=128)
    instrument_selector = InstrumentSelector(temperature_constant=1.0)
    strategy_finder = GraphStrategyFinder()
    orchestrator = ExecutionOrchestrator(
        embedding_service,
        instrument_selector,
        strategy_finder
    )
    critic = SimpleCritic()
    feedback_collector = FeedbackCollector()
    training_orchestrator = TrainingOrchestrator(learning_rate=0.01)
    
    print("Компоненты инициализированы\n")
    
    # Создание тестовых инструментов
    print("Создание инструментов...")
    tools = create_test_tools(embedding_service)
    print(f"Создано инструментов: {len(tools)}\n")
    
    # Вывод списка инструментов
    print("Список доступных инструментов:")
    for i, tool in enumerate(tools):
        print(f"  {i + 1}. {tool.metadata.tool_name:<25} "
              f"[{tool.input.format} -> {tool.output.format}] "
              f"Reputation: {tool.reputation:.2f}")
    print()
    
    # Определение задачи
    task = TaskDefinition(
        description="Ответить на вопрос пользователя",
        input_connector=Connector(data_format="text", semantic_format="question"),
        output_connector=Connector(data_format="text", semantic_format="answer"),
        input_data="Какая сегодня погода?",
        domain="qa"
    )
    
    print("Задача:")
    print(f"  Описание: {task.description}")
    print(f"  Вход:     {task.input_connector.format}")
    print(f"  Выход:    {task.output_connector.format}")
    print(f"  Данные:   {task.input_data}")
    print()
    
    # Выбор алгоритма поиска пути
    algorithm = select_algorithm()
    print()
    
    # Поиск стратегии
    print(f"Поиск пути с использованием алгоритма: {get_algorithm_name(algorithm)}")
    
    path_limit = 1 if algorithm in [PathfindingAlgorithm.DIJKSTRA, PathfindingAlgorithm.ASTAR] else 3
    
    strategies = strategy_finder.find_strategies(
        tools,
        task.input_connector.format,
        task.output_connector.format,
        path_limit,
        algorithm
    )
    
    if not strategies:
        print("ОШИБКА: Пути не найдены!")
        return
    
    print(f"Найдено путей: {len(strategies)}\n")
    
    # Вывод найденных путей
    for i, strategy in enumerate(strategies):
        total_weight = sum(t.get_graph_weight() for t in strategy)
        
        print(f"Путь {i + 1}: (вес: {total_weight:.3f}, шагов: {len(strategy)})")
        for j, tool in enumerate(strategy):
            arrow = " -> " if j < len(strategy) - 1 else ""
            print(f"  Шаг {j + 1}: {tool.metadata.tool_name:<25} "
                  f"[{tool.input.format} -> {tool.output.format}]{arrow}")
        print()
    
    # Выполнение первого пути
    print("=" * 70)
    print("Выполнение выбранной стратегии...\n")
    
    task.task_embedding = embedding_service.embed_text(task.description)
    
    context = orchestrator.execute_task(task, tools, path_limit=1, top_k=3)
    
    # Вывод результатов выполнения
    print("РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ:")
    print("-" * 70)
    print(f"Статус:           {context.status.value}")
    print(f"Время выполнения: {context.total_time:.3f} сек")
    print(f"Стоимость:        {context.total_cost:.2f}")
    print(f"Шагов выполнено:  {len(context.execution_steps)}")
    print()
    
    # Детали каждого шага
    print("Детали выполнения:")
    for step in context.execution_steps:
        print(f"\n  Шаг {step.step_number}:")
        print(f"    Инструмент:         {step.selected_tool.metadata.tool_name}")
        print(f"    Время выполнения:   {step.execution_time:.3f} сек")
        print(f"    Стоимость:          {step.cost:.2f}")
        print(f"    Успех:              {'Да' if step.success else 'Нет'}")
        
        if step.selection_result:
            print(f"    Вероятность выбора: {step.selection_result.selection_probability:.3f}")
            print(f"    Температура:        {step.selection_result.temperature:.3f}")
            
            if len(step.selection_result.all_probabilities) > 1:
                print(f"    Кандидаты (top-{step.selection_result.top_k}):")
                
                # Сортируем по вероятности
                sorted_probs = sorted(
                    step.selection_result.all_probabilities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for tool, prob in sorted_probs:
                    logit = step.selection_result.all_logits[tool]
                    mark = "*" if tool == step.selected_tool else " "
                    print(f"      {mark} {tool.metadata.tool_name:<20} "
                          f"logit: {logit:6.2f}, P: {prob:.3f}")
    print()
    
    if context.result:
        print(f"РЕЗУЛЬТАТ: {context.result}\n")
    
    # Обратная связь и обучение
    print("=" * 70)
    print("Обратная связь и обучение:\n")
    
    critic_feedback = critic.evaluate_execution(context)
    feedback_collector.add_feedback(critic_feedback)
    print(f"  Оценка критика:      {critic_feedback.quality_score:.2f}")
    
    user_feedback = FeedbackData(
        task_id=context.task_id,
        source=FeedbackSource.USER,
        success=True,
        quality_score=0.85,
        comment="Хороший ответ"
    )
    feedback_collector.add_feedback(user_feedback)
    print(f"  Оценка пользователя: {user_feedback.quality_score:.2f}")
    
    aggregated_score = feedback_collector.get_average_quality(context.task_id)
    print(f"  Итоговая оценка:     {aggregated_score:.2f}")
    print()
    
    # Обучение инструментов
    print("Статистика инструментов ДО обучения:")
    used_tools = list({step.selected_tool for step in context.execution_steps})
    for tool in used_tools:
        print(f"  {tool.metadata.tool_name:<25} "
              f"Reputation: {tool.reputation:.3f}, "
              f"Samples: {tool.metadata.training_sample_size}")
    
    all_feedbacks = feedback_collector.get_feedbacks(context.task_id)
    training_orchestrator.add_execution_to_dataset(context, all_feedbacks)
    training_orchestrator.train_all_tools(tools)
    training_orchestrator.update_tool_embeddings(tools)
    
    print()
    print("Статистика инструментов ПОСЛЕ обучения:")
    for tool in used_tools:
        print(f"  {tool.metadata.tool_name:<25} "
              f"Reputation: {tool.reputation:.3f}, "
              f"Samples: {tool.metadata.training_sample_size}")
    
    stats = training_orchestrator.get_statistics()
    print()
    print("Общая статистика обучения:")
    print(f"  Выполнений:        {stats.total_executions}")
    print(f"  Среднее качество:  {stats.average_quality:.2f}")
    print(f"  Success Rate:      {stats.success_rate:.0%}")
    print(f"  Среднее время:     {stats.average_execution_time:.2f} сек")
    print(f"  Средняя стоимость: {stats.average_cost:.2f}")
    print()
    
    print("Обучение завершено.")


def select_algorithm() -> PathfindingAlgorithm:
    """Интерактивный выбор алгоритма поиска пути"""
    
    print("=" * 70)
    print("ВЫБОР АЛГОРИТМА ПОИСКА ПУТИ:\n")
    print("  1. Дейкстра        - Быстрый поиск одного оптимального пути")
    print("  2. Йена            - Поиск топ-K альтернативных путей")
    print("  3. A-star          - Поиск с эвристикой (один путь)")
    print("  4. Муравьиный      - Стохастический поиск топ-K путей")
    print()
    
    while True:
        try:
            choice = input("Введите номер алгоритма (1-4): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= 4:
                return {
                    1: PathfindingAlgorithm.DIJKSTRA,
                    2: PathfindingAlgorithm.YEN,
                    3: PathfindingAlgorithm.ASTAR,
                    4: PathfindingAlgorithm.ANT_COLONY
                }[choice_num]
            else:
                print("Ошибка! Введите число от 1 до 4")
        except ValueError:
            print("Ошибка! Введите число от 1 до 4")
        except KeyboardInterrupt:
            print("\nПрервано пользователем")
            sys.exit(0)


def get_algorithm_name(algorithm: PathfindingAlgorithm) -> str:
    """Получить название алгоритма"""
    
    names = {
        PathfindingAlgorithm.DIJKSTRA: "Дейкстра",
        PathfindingAlgorithm.YEN: "Йена",
        PathfindingAlgorithm.ASTAR: "A-star",
        PathfindingAlgorithm.ANT_COLONY: "Муравьиный алгоритм"
    }
    
    return names.get(algorithm, "Неизвестный")


def create_test_tools(embedding_service) -> List[BaseTool]:
    """Создание тестовых инструментов для демонстрации"""
    
    tools = []
    
    # Инструменты-парсеры вопросов
    parser1 = ReactiveTool(
        "text", "question",
        "text", "parsed_question",
        "QuestionParser_v1",
        lambda input_data: f"[Parsed v1] {input_data}"
    )
    parser1.reputation = 0.75
    parser1.metadata.capabilities_embedding = embedding_service.embed_tool_capabilities(parser1)
    tools.append(parser1)
    
    parser2 = ReactiveTool(
        "text", "question",
        "text", "parsed_question",
        "QuestionParser_v2",
        lambda input_data: f"[Parsed v2] {input_data}"
    )
    parser2.reputation = 0.65
    parser2.metadata.capabilities_embedding = embedding_service.embed_tool_capabilities(parser2)
    tools.append(parser2)
    
    # Инструменты для генерации ответов
    qa_gpt = ReactiveTool(
        "text", "parsed_question",
        "text", "answer",
        "QA_GPT4",
        lambda input_data: "[GPT-4] Температура в Москве +15C, облачно"
    )
    qa_gpt.reputation = 0.85
    qa_gpt.mean_cost = 0.5
    qa_gpt.metadata.capabilities_embedding = embedding_service.embed_tool_capabilities(qa_gpt)
    tools.append(qa_gpt)
    
    qa_claude = ReactiveTool(
        "text", "parsed_question",
        "text", "answer",
        "QA_Claude3",
        lambda input_data: "[Claude-3] Москва: +15C, переменная облачность"
    )
    qa_claude.reputation = 0.80
    qa_claude.mean_cost = 0.4
    qa_claude.metadata.capabilities_embedding = embedding_service.embed_tool_capabilities(qa_claude)
    tools.append(qa_claude)
    
    qa_gemini = ReactiveTool(
        "text", "parsed_question",
        "text", "answer",
        "QA_Gemini",
        lambda input_data: "[Gemini] Погода: +15C, облачно"
    )
    qa_gemini.reputation = 0.75
    qa_gemini.mean_cost = 0.3
    qa_gemini.metadata.capabilities_embedding = embedding_service.embed_tool_capabilities(qa_gemini)
    tools.append(qa_gemini)
    
    # Прямой инструмент (обходной путь)
    direct_qa = ReactiveTool(
        "text", "question",
        "text", "answer",
        "Direct_QA_Simple",
        lambda input_data: "[Simple] Погода хорошая, около +15C"
    )
    direct_qa.reputation = 0.60
    direct_qa.mean_cost = 0.1
    direct_qa.metadata.capabilities_embedding = embedding_service.embed_tool_capabilities(direct_qa)
    tools.append(direct_qa)
    
    return tools


if __name__ == "__main__":
    main()
