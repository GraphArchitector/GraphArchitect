"""Оркестратор выполнения стратегии решения задачи"""

from typing import List, Optional
from datetime import datetime

from ...entities.base_tool import BaseTool
from ...entities.task_definition import TaskDefinition
from ..selection.instrument_selector import InstrumentSelector
from ..graph_strategy_finder import GraphStrategyFinder
from ..tool_edge import ToolEdge
from .execution_context import ExecutionContext
from .execution_status import ExecutionStatus
from .execution_step import ExecutionStep
from ..pathfinding_algorithm import PathfindingAlgorithm

# Используем TYPE_CHECKING для избежания циклических импортов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..embedding.embedding_service import EmbeddingService


class ExecutionOrchestrator:
    """
    Оркестратор выполнения задачи.
    
    Координирует:
    1. Поиск стратегии (цепочки инструментов)
    2. Выполнение каждого шага с выбором инструмента
    3. Сохранение градиентных трасс для обучения
    """
    
    def __init__(
        self,
        embedding_service: 'EmbeddingService',
        instrument_selector: InstrumentSelector,
        strategy_finder: GraphStrategyFinder
    ):
        """
        Инициализация оркестратора.
        
        Args:
            embedding_service: Сервис векторизации
            instrument_selector: Селектор инструментов
            strategy_finder: Поиск стратегий в графе
        """
        self._embedding_service = embedding_service
        self._instrument_selector = instrument_selector
        self._strategy_finder = strategy_finder
    
    def execute_task(
        self,
        task: TaskDefinition,
        available_tools: List[BaseTool],
        path_limit: int = 1,
        top_k: int = 5,
        algorithm: Optional[PathfindingAlgorithm] = None
    ) -> ExecutionContext:
        """
        Выполнить задачу: найти стратегию и выполнить цепочку инструментов.
        
        Args:
            task: Определение задачи
            available_tools: Доступные инструменты
            path_limit: Количество путей для поиска
            top_k: Количество лучших кандидатов при выборе
            
        Returns:
            Контекст выполнения с результатами
        """
        context = ExecutionContext(
            task_id=task.task_id,
            task=task,
            input_data=task.input_data,
            current_data=task.input_data,
            start_time=datetime.utcnow(),
            status=ExecutionStatus.RUNNING
        )
        
        try:
            # 1. Создать эмбеддинг задачи если его нет
            if task.task_embedding is None and task.description:
                task.task_embedding = self._embedding_service.embed_text(
                    task.description
                )
            
            # 2. Найти стратегию (цепочку групп инструментов)
            start_format = task.input_connector.format
            end_format = task.output_connector.format
            
            strategies = self._strategy_finder.find_strategies(
                available_tools,
                start_format,
                end_format,
                path_limit,
                algorithm
            )
            
            if not strategies:
                context.status = ExecutionStatus.FAILED
                context.error_message = (
                    "Не найдено путей от входного к выходному коннектору"
                )
                return context
            
            # 3. Берем первую стратегию
            strategy = strategies[0]
            
            # 4. Выполнить цепочку
            self._execute_strategy(context, strategy, task.task_embedding, top_k)
            
            # 5. Установить финальный статус
            context.status = ExecutionStatus.COMPLETED
            context.result = context.current_data
            
        except Exception as e:
            context.status = ExecutionStatus.FAILED
            context.error_message = str(e)
        
        finally:
            context.end_time = datetime.utcnow()
            context.total_time = (
                context.end_time - context.start_time
            ).total_seconds()
        
        return context
    
    def execute_task_with_edges(
        self,
        task: TaskDefinition,
        strategy_path: List[ToolEdge],
        top_k: int = 5
    ) -> ExecutionContext:
        """
        Выполнить задачу с использованием готовых рёбер графа.
        
        Args:
            task: Определение задачи
            strategy_path: Путь в графе (список рёбер)
            top_k: Количество лучших кандидатов при выборе
            
        Returns:
            Контекст выполнения с результатами
        """
        context = ExecutionContext(
            task_id=task.task_id,
            task=task,
            input_data=task.input_data,
            current_data=task.input_data,
            start_time=datetime.utcnow(),
            status=ExecutionStatus.RUNNING
        )
        
        try:
            # Создать эмбеддинг задачи если его нет
            if task.task_embedding is None and task.description:
                task.task_embedding = self._embedding_service.embed_text(
                    task.description
                )
            
            # Выполнить цепочку рёбер
            for i, edge in enumerate(strategy_path):
                self._execute_instrument_group(
                    context,
                    edge.tools,
                    task.task_embedding,
                    i + 1,
                    top_k
                )
                
                if not context.execution_steps[-1].success:
                    context.status = ExecutionStatus.FAILED
                    break
            
            if context.status == ExecutionStatus.RUNNING:
                context.status = ExecutionStatus.COMPLETED
                context.result = context.current_data
            
        except Exception as e:
            context.status = ExecutionStatus.FAILED
            context.error_message = str(e)
        
        finally:
            context.end_time = datetime.utcnow()
            context.total_time = (
                context.end_time - context.start_time
            ).total_seconds()
        
        return context
    
    def _execute_strategy(
        self,
        context: ExecutionContext,
        strategy: List[BaseTool],
        task_embedding: Optional[List[float]],
        top_k: int
    ):
        """
        Выполнить стратегию - последовательность групп инструментов.
        
        В текущей реализации strategy - это уже выбранные инструменты.
        Для полной реализации нужно получить группу инструментов на ребре.
        """
        for i, tool in enumerate(strategy):
            # Упрощение: используем один инструмент как группу из одного элемента
            tools = [tool]
            self._execute_instrument_group(
                context,
                tools,
                task_embedding,
                i + 1,
                top_k
            )
            
            if not context.execution_steps[-1].success:
                context.status = ExecutionStatus.FAILED
                break
    
    def _execute_instrument_group(
        self,
        context: ExecutionContext,
        tools: List[BaseTool],
        task_embedding: Optional[List[float]],
        step_number: int,
        top_k: int
    ):
        """
        Выполнить выбор и запуск инструмента из группы.
        
        Это ключевой метод, реализующий пункт 4 из описания системы:
        1. Формируется векторное представление задачи
        2. Этот вектор отправляется всем инструментам в группе
        3. Получаем логиты от всех инструментов
        4. Отбираем топ-K инструментов
        5. Строим распределение вероятностей через softmax с температурой
        6. Выбираем инструмент через сэмплирование
        7. Сохраняем градиентную информацию
        """
        step = ExecutionStep(
            step_number=step_number,
            available_tools=tools,
            input_data=context.current_data,
            start_time=datetime.utcnow()
        )
        
        try:
            # 1. Выбрать инструмент с помощью логитов и температуры
            step.selection_result = self._instrument_selector.select_instrument(
                tools,
                task_embedding,
                top_k
            )
            
            step.selected_tool = step.selection_result.selected_tool
            
            # 2. Сохранить градиентную информацию
            context.add_gradient_trace(step.selection_result.gradient_info)
            
            # 3. Выполнить инструмент
            step.output_data = step.selected_tool.execute(step.input_data)
            step.success = True
            
            # 4. Обновить текущие данные контекста
            context.current_data = step.output_data
            
            # 5. Записать метрики
            step.end_time = datetime.utcnow()
            step.execution_time = (
                step.end_time - step.start_time
            ).total_seconds()
            step.cost = step.selected_tool.mean_cost
            
            context.total_cost += step.cost
            
        except Exception as e:
            step.success = False
            step.error_message = str(e)
            step.end_time = datetime.utcnow()
        
        context.add_step(step)
