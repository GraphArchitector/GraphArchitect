import logging
import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import statistics

from ..execution.execution_context import ExecutionContext
from ..feedback.feedback_data import FeedbackData, FeedbackSource
from ..training.training_orchestrator import TrainingOrchestrator
from .llm_critic import LLMCritic, LLMCriticScore
from ...entities.base_tool import BaseTool

logger = logging.getLogger(__name__)


class UpdateStrategy(Enum):
    """Стратегия обновления репутации инструментов."""
    POLICY_GRADIENT = "policy_gradient"
    EXPONENTIAL_MOVING_AVERAGE = "exponential_moving_average"
    DIRECT_ASSIGNMENT = "direct_assignment"


@dataclass
class ToolUpdateRecord:
    """Запись об обновлении инструмента."""
    tool_name: str
    old_reputation: float
    new_reputation: float
    delta_reputation: float
    reward: float
    learning_rate: float
    feedback_quality: float
    success: bool
    timestamp: str
    update_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RLAIFTrainingResult:
    """Результат RLAIF обучения."""
    training_id: str
    evaluations_count: int
    successful_count: int
    failed_count: int
    average_score: float
    average_correctness: float
    average_completeness: float
    average_relevance: float
    average_clarity: float
    tools_updated: int
    improvements: Dict[str, float]
    tool_updates: List[ToolUpdateRecord]
    timestamp: str
    duration_seconds: float


@dataclass
class RLAIFTrainingConfig:
    """Конфигурация RLAIF обучения."""
    min_score_threshold: float = 0.3
    learning_rate: float = 0.01
    update_strategy: UpdateStrategy = UpdateStrategy.POLICY_GRADIENT
    ema_alpha: float = 0.1
    save_evaluations: bool = True
    save_tool_updates: bool = True
    log_level: str = "INFO"
    seed: Optional[int] = None
    max_reputation_history: int = 1000
    convergence_window_size: int = 5
    convergence_tolerance: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация конфигурации."""
        return {
            'min_score_threshold': self.min_score_threshold,
            'learning_rate': self.learning_rate,
            'update_strategy': self.update_strategy.value,
            'ema_alpha': self.ema_alpha,
            'save_evaluations': self.save_evaluations,
            'save_tool_updates': self.save_tool_updates,
            'log_level': self.log_level,
            'seed': self.seed,
            'max_reputation_history': self.max_reputation_history,
            'convergence_window_size': self.convergence_window_size,
            'convergence_tolerance': self.convergence_tolerance
        }


class RLAIFTrainer:
    """
    Тренер с использованием RLAIF (Reinforcement Learning from AI Feedback).
    
    Автоматически выполняет следующие операции:
    1. Оценивает результаты через LLM критика
    2. Создает FeedbackData на основе оценок
    3. Обновляет репутацию инструментов используя выбранную стратегию
    4. Сохраняет историю для анализа и воспроизводимости
    
    Параметры:
        llm_critic: LLM критик для оценки качества ответов
        training_orchestrator: Оркестратор обучения
        config: Конфигурация обучения
    """
    
    def __init__(
        self,
        llm_critic: LLMCritic,
        training_orchestrator: TrainingOrchestrator,
        config: Optional[RLAIFTrainingConfig] = None
    ):
        """
        Инициализация RLAIF тренера.
        
        Аргументы:
            llm_critic: LLM критик для оценки
            training_orchestrator: Оркестратор обучения
            config: Конфигурация (если None, используются значения по умолчанию)
        """
        self._config = config or RLAIFTrainingConfig()
        self._critic = llm_critic
        self._training = training_orchestrator
        
        # История оценок
        self._evaluation_history: List[LLMCriticScore] = []
        
        # История обновлений инструментов
        self._tool_update_history: List[ToolUpdateRecord] = []
        
        # История обучения по итерациям
        self._training_iterations: List[Dict[str, Any]] = []
        
        # Репутация инструментов во времени (для анализа)
        self._reputation_timeline: Dict[str, List[Tuple[str, float]]] = {}
        
        # Статистика по инструментам
        self._tool_statistics: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"RLAIF Trainer initialized with config: {self._config.to_dict()}"
        )
    
    def evaluate_and_train(
        self,
        context: ExecutionContext,
        task_description: str,
        result: str,
        iteration_num: Optional[int] = None
    ) -> Optional[RLAIFTrainingResult]:
        """
        Оценить результат и обучить инструменты.
        
        Процесс:
        1. Оценка результата через LLM критика
        2. Валидация оценки
        3. Создание FeedbackData
        4. Обновление репутации инструментов
        5. Сохранение метрик
        
        Аргументы:
            context: Контекст выполнения задачи
            task_description: Описание задачи
            result: Результат выполнения
            iteration_num: Номер итерации обучения (опционально)
            
        Возвращает:
            RLAIFTrainingResult или None при ошибке
        """
        training_id = str(uuid.uuid4())
        timestamp_start = datetime.utcnow()
        
        logger.info(
            f"Starting RLAIF training: training_id={training_id}, "
            f"iteration={iteration_num}"
        )
        
        try:
            # Валидация входных данных
            if not self._validate_inputs(context, task_description, result):
                return None
            
            # Шаг 1: Оценка через LLM критика
            score = self._evaluate_with_critic(
                task_description,
                result,
                context
            )
            
            if score is None:
                logger.error("LLM Critic evaluation failed")
                return None
            
            # Сохранение оценки
            if self._config.save_evaluations:
                self._evaluation_history.append(score)
            
            # Шаг 2: Создание FeedbackData
            feedback = self._create_feedback(context, score)
            
            # Шаг 3: Обновление инструментов
            tool_updates = self._update_tools(
                context,
                feedback,
                training_id
            )
            
            # Шаг 4: Добавление в датасет обучения
            self._training.add_execution_to_dataset(context, [feedback])
            
            # Сохранение истории обновлений
            if self._config.save_tool_updates:
                self._tool_update_history.extend(tool_updates)
            
            # Вычисление длительности
            timestamp_end = datetime.utcnow()
            duration = (timestamp_end - timestamp_start).total_seconds()
            
            # Создание результата
            result_obj = RLAIFTrainingResult(
                training_id=training_id,
                evaluations_count=1,
                successful_count=1,
                failed_count=0,
                average_score=score.overall_score,
                average_correctness=score.correctness,
                average_completeness=score.completeness,
                average_relevance=score.relevance,
                average_clarity=score.clarity,
                tools_updated=len(tool_updates),
                improvements={
                    update.tool_name: update.delta_reputation
                    for update in tool_updates
                },
                tool_updates=tool_updates,
                timestamp=timestamp_start.isoformat(),
                duration_seconds=duration
            )
            
            # Сохранение в историю итераций
            self._training_iterations.append({
                'iteration': iteration_num,
                'training_id': training_id,
                'timestamp': timestamp_start.isoformat(),
                'average_score': score.overall_score,
                'tools_updated': len(tool_updates)
            })
            
            logger.info(
                f"RLAIF training completed: score={score.overall_score:.4f}, "
                f"tools_updated={len(tool_updates)}, duration={duration:.2f}s"
            )
            
            return result_obj
        
        except Exception as error:
            logger.error(f"RLAIF training failed: {error}", exc_info=True)
            return None
    
    def _validate_inputs(
        self,
        context: ExecutionContext,
        task_description: str,
        result: str
    ) -> bool:
        """
        Валидировать входные данные.
        
        Аргументы:
            context: Контекст выполнения
            task_description: Описание задачи
            result: Результат
            
        Возвращает:
            True если данные валидны
        """
        if not context:
            logger.error("Invalid input: context is None")
            return False
        
        if not task_description or not isinstance(task_description, str):
            logger.error("Invalid input: task_description is empty or not string")
            return False
        
        if not result or not isinstance(result, str):
            logger.error("Invalid input: result is empty or not string")
            return False
        
        if not context.execution_steps:
            logger.warning("Warning: execution_steps is empty")
        
        return True
    
    def _evaluate_with_critic(
        self,
        task_description: str,
        result: str,
        context: ExecutionContext
    ) -> Optional[LLMCriticScore]:
        """
        Оценить результат используя LLM критика.
        
        Аргументы:
            task_description: Описание задачи
            result: Результат выполнения
            context: Контекст выполнения
            
        Возвращает:
            LLMCriticScore или None
        """
        try:
            critic_context = {
                'execution_time': context.total_time if hasattr(context, 'total_time') else 0.0,
                'tools_used': [
                    step.selected_tool.metadata.tool_name
                    for step in context.execution_steps
                    if hasattr(step, 'selected_tool') and step.selected_tool
                ] if context.execution_steps else [],
                'cost': context.total_cost if hasattr(context, 'total_cost') else 0.0,
                'iterations': len(context.execution_steps) if context.execution_steps else 0
            }
            
            score = self._critic.evaluate_answer(
                task=task_description,
                answer=result,
                context=critic_context
            )
            
            if score is None:
                logger.error("LLM Critic returned None")
                return None
            
            # Валидация оценки
            if not self._validate_score(score):
                return None
            
            logger.info(
                f"LLM Critic evaluation: "
                f"overall={score.overall_score:.4f}, "
                f"correctness={score.correctness:.4f}, "
                f"completeness={score.completeness:.4f}, "
                f"relevance={score.relevance:.4f}, "
                f"clarity={score.clarity:.4f}"
            )
            
            return score
        
        except Exception as error:
            logger.error(f"LLM Critic evaluation failed: {error}")
            return None
    
    def _validate_score(self, score: LLMCriticScore) -> bool:
        """
        Валидировать оценку от критика.
        
        Аргументы:
            score: Оценка для валидации
            
        Возвращает:
            True если оценка валидна
        """
        def is_valid_score_value(value) -> bool:
            """Проверить что значение в диапазоне 0-1."""
            return isinstance(value, (int, float)) and 0.0 <= value <= 1.0
        
        if not is_valid_score_value(score.overall_score):
            logger.error(
                f"Invalid overall_score: {score.overall_score} "
                f"(must be 0.0-1.0)"
            )
            return False
        
        if not is_valid_score_value(score.correctness):
            logger.error(
                f"Invalid correctness: {score.correctness} (must be 0.0-1.0)"
            )
            return False
        
        if not is_valid_score_value(score.completeness):
            logger.error(
                f"Invalid completeness: {score.completeness} (must be 0.0-1.0)"
            )
            return False
        
        if not is_valid_score_value(score.relevance):
            logger.error(f"Invalid relevance: {score.relevance} (must be 0.0-1.0)")
            return False
        
        if not is_valid_score_value(score.clarity):
            logger.error(f"Invalid clarity: {score.clarity} (must be 0.0-1.0)")
            return False
        
        return True
    
    def _create_feedback(
        self,
        context: ExecutionContext,
        score: LLMCriticScore
    ) -> FeedbackData:
        """
        Создать FeedbackData на основе оценки.
        
        Аргументы:
            context: Контекст выполнения
            score: Оценка от критика
            
        Возвращает:
            FeedbackData
        """
        feedback = FeedbackData(
            task_id=getattr(context, 'task_id', str(uuid.uuid4())),
            source=FeedbackSource.AUTO_CRITIC,
            quality_score=score.overall_score,
            comment=score.reasoning,
            detailed_scores={
                'correctness': score.correctness,
                'completeness': score.completeness,
                'relevance': score.relevance,
                'clarity': score.clarity,
                'overall': score.overall_score
            },
            success=score.overall_score >= self._config.min_score_threshold,
            timestamp=score.timestamp,
            critic_model=score.model_used,
            evaluation_id=score.evaluation_id
        )
        
        return feedback
    
    def _update_tools(
        self,
        context: ExecutionContext,
        feedback: FeedbackData,
        training_id: str
    ) -> List[ToolUpdateRecord]:
        """
        Обновить репутацию инструментов используя выбранную стратегию.
        
        Аргументы:
            context: Контекст выполнения
            feedback: Обратная связь
            training_id: ID сессии обучения
            
        Возвращает:
            Список ToolUpdateRecord
        """
        updates = []
        
        if not context.execution_steps:
            logger.warning("No execution steps to update")
            return updates
        
        tools_to_train = [
            step.selected_tool
            for step in context.execution_steps
            if hasattr(step, 'selected_tool') and step.selected_tool
        ]
        
        logger.debug(f"Updating {len(tools_to_train)} tools")
        
        for tool in tools_to_train:
            try:
                update = self._update_single_tool(tool, feedback)
                if update:
                    updates.append(update)
                    
                    # Сохранение в timeline
                    self._record_reputation_change(tool.metadata.tool_name, update)
                    
                    # Обновление статистики
                    self._update_tool_statistics(tool.metadata.tool_name, update)
            
            except Exception as error:
                logger.error(
                    f"Failed to update tool {tool.metadata.tool_name}: {error}"
                )
                continue
        
        return updates
    
    def _update_single_tool(
        self,
        tool: BaseTool,
        feedback: FeedbackData
    ) -> Optional[ToolUpdateRecord]:
        """
        Обновить один инструмент.
        
        Аргументы:
            tool: Инструмент для обновления
            feedback: Обратная связь
            
        Возвращает:
            ToolUpdateRecord или None
        """
        old_reputation = getattr(tool.metadata, 'reputation', 0.5)
        reward = feedback.quality_score
        
        # Выбор стратегии обновления
        if self._config.update_strategy == UpdateStrategy.POLICY_GRADIENT:
            new_reputation = self._policy_gradient_update(
                old_reputation,
                reward
            )
        
        elif self._config.update_strategy == UpdateStrategy.EXPONENTIAL_MOVING_AVERAGE:
            new_reputation = self._ema_update(
                old_reputation,
                reward
            )
        
        elif self._config.update_strategy == UpdateStrategy.DIRECT_ASSIGNMENT:
            new_reputation = reward
        
        else:
            logger.error(f"Unknown update strategy: {self._config.update_strategy}")
            return None
        
        # Нормализация репутации в диапазон [0, 1]
        new_reputation = max(0.0, min(1.0, new_reputation))
        
        # Обновление инструмента
        tool.metadata.reputation = new_reputation
        
        delta = new_reputation - old_reputation
        
        update = ToolUpdateRecord(
            tool_name=tool.metadata.tool_name,
            old_reputation=old_reputation,
            new_reputation=new_reputation,
            delta_reputation=delta,
            reward=reward,
            learning_rate=self._config.learning_rate,
            feedback_quality=feedback.quality_score,
            success=feedback.success,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.debug(
            f"Tool update: {tool.metadata.tool_name} "
            f"reputation {old_reputation:.4f} -> {new_reputation:.4f} "
            f"(delta={delta:+.6f}, reward={reward:.4f}, "
            f"strategy={self._config.update_strategy.value})"
        )
        
        return update
    
    def _policy_gradient_update(
        self,
        current_reputation: float,
        reward: float
    ) -> float:
        """
        Обновление репутации используя Policy Gradient.
        
        Формула: new_reputation = current + learning_rate * reward
        
        Аргументы:
            current_reputation: Текущая репутация
            reward: Награда (quality_score)
            
        Возвращает:
            Новая репутация
        """
        delta = self._config.learning_rate * reward
        new_reputation = current_reputation + delta
        
        return new_reputation
    
    def _ema_update(
        self,
        current_reputation: float,
        new_value: float
    ) -> float:
        """
        Обновление репутации используя Exponential Moving Average.
        
        Формула: new = alpha * new_value + (1 - alpha) * current
        
        Аргументы:
            current_reputation: Текущая репутация
            new_value: Новое значение (quality_score)
            
        Возвращает:
            Новая репутация
        """
        new_reputation = (
            self._config.ema_alpha * new_value +
            (1 - self._config.ema_alpha) * current_reputation
        )
        
        return new_reputation
    
    def _record_reputation_change(
        self,
        tool_name: str,
        update: ToolUpdateRecord
    ):
        """
        Записать изменение репутации в timeline.
        
        Аргументы:
            tool_name: Имя инструмента
            update: Запись об обновлении
        """
        if tool_name not in self._reputation_timeline:
            self._reputation_timeline[tool_name] = []
        
        self._reputation_timeline[tool_name].append(
            (update.timestamp, update.new_reputation)
        )
        
        # Ограничение размера истории
        if len(self._reputation_timeline[tool_name]) > self._config.max_reputation_history:
            self._reputation_timeline[tool_name] = (
                self._reputation_timeline[tool_name][-self._config.max_reputation_history:]
            )
    
    def _update_tool_statistics(
        self,
        tool_name: str,
        update: ToolUpdateRecord
    ):
        """
        Обновить статистику инструмента.
        
        Аргументы:
            tool_name: Имя инструмента
            update: Запись об обновлении
        """
        if tool_name not in self._tool_statistics:
            self._tool_statistics[tool_name] = {
                'update_count': 0,
                'total_delta': 0.0,
                'avg_delta': 0.0,
                'max_reputation': 0.0,
                'min_reputation': 1.0,
                'success_count': 0,
                'failure_count': 0
            }
        
        stats = self._tool_statistics[tool_name]
        stats['update_count'] += 1
        stats['total_delta'] += update.delta_reputation
        stats['avg_delta'] = stats['total_delta'] / stats['update_count']
        stats['max_reputation'] = max(stats['max_reputation'], update.new_reputation)
        stats['min_reputation'] = min(stats['min_reputation'], update.new_reputation)
        
        if update.success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1
    
    def batch_evaluate_and_train(
        self,
        executions: List[Tuple[ExecutionContext, str, str]],
        start_iteration: int = 1
    ) -> Optional[RLAIFTrainingResult]:
        """
        Batch оценка и обучение на нескольких выполнениях.
        
        Аргументы:
            executions: Список кортежей (context, task_description, result)
            start_iteration: Начальный номер итерации
            
        Возвращает:
            Агрегированный RLAIFTrainingResult или None
        """
        if not executions:
            logger.warning("Empty executions list for batch training")
            return None
        
        logger.info(f"Starting batch training: {len(executions)} executions")
        
        batch_id = str(uuid.uuid4())
        timestamp_start = datetime.utcnow()
        
        total_score = 0.0
        total_correctness = 0.0
        total_completeness = 0.0
        total_relevance = 0.0
        total_clarity = 0.0
        
        total_updated = 0
        all_improvements = {}
        all_tool_updates = []
        
        successful_count = 0
        failed_count = 0
        
        for idx, execution in enumerate(executions):
            try:
                if len(execution) != 3:
                    logger.warning(
                        f"Invalid execution format at index {idx}: "
                        f"expected 3 items, got {len(execution)}"
                    )
                    failed_count += 1
                    continue
                
                context, task, result = execution
                iteration_num = start_iteration + idx
                
                train_result = self.evaluate_and_train(
                    context,
                    task,
                    result,
                    iteration_num=iteration_num
                )
                
                if train_result is None:
                    failed_count += 1
                    logger.warning(f"Execution {idx} (iteration {iteration_num}) returned None")
                    continue
                
                total_score += train_result.average_score
                total_correctness += train_result.average_correctness
                total_completeness += train_result.average_completeness
                total_relevance += train_result.average_relevance
                total_clarity += train_result.average_clarity
                
                total_updated += train_result.tools_updated
                all_tool_updates.extend(train_result.tool_updates)
                successful_count += 1
                
                for tool_name, delta in train_result.improvements.items():
                    if tool_name not in all_improvements:
                        all_improvements[tool_name] = []
                    all_improvements[tool_name].append(delta)
            
            except Exception as error:
                failed_count += 1
                logger.error(f"Failed to process execution {idx}: {error}", exc_info=True)
                continue
        
        if successful_count == 0:
            logger.error("No successful executions in batch")
            return None
        
        # Агрегация результатов
        avg_score = total_score / successful_count
        avg_correctness = total_correctness / successful_count
        avg_completeness = total_completeness / successful_count
        avg_relevance = total_relevance / successful_count
        avg_clarity = total_clarity / successful_count
        
        avg_improvements = {
            tool_name: sum(deltas) / len(deltas)
            for tool_name, deltas in all_improvements.items()
        }
        
        timestamp_end = datetime.utcnow()
        duration = (timestamp_end - timestamp_start).total_seconds()
        
        result = RLAIFTrainingResult(
            training_id=batch_id,
            evaluations_count=len(executions),
            successful_count=successful_count,
            failed_count=failed_count,
            average_score=avg_score,
            average_correctness=avg_correctness,
            average_completeness=avg_completeness,
            average_relevance=avg_relevance,
            average_clarity=avg_clarity,
            tools_updated=total_updated,
            improvements=avg_improvements,
            tool_updates=all_tool_updates,
            timestamp=timestamp_start.isoformat(),
            duration_seconds=duration
        )
        
        logger.info(
            f"Batch training completed: batch_id={batch_id}, "
            f"successful={successful_count}, failed={failed_count}, "
            f"avg_score={avg_score:.4f}, tools_updated={total_updated}, "
            f"duration={duration:.2f}s"
        )
        
        return result
    
    def check_convergence(
        self,
        window_size: Optional[int] = None,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        Проверить сошлась ли система обучения.
        
        Сходимость определяется как максимальное изменение scores в окне
        меньшее чем tolerance.
        
        Аргументы:
            window_size: Размер окна (если None, используется из config)
            tolerance: Допустимое изменение (если None, используется из config)
            
        Возвращает:
            True если система сошлась
        """
        window_size = window_size or self._config.convergence_window_size
        tolerance = tolerance or self._config.convergence_tolerance
        
        if len(self._evaluation_history) < window_size:
            logger.debug(
                f"Not enough evaluation history for convergence check: "
                f"{len(self._evaluation_history)} < {window_size}"
            )
            return False
        
        recent_scores = [
            s.overall_score
            for s in self._evaluation_history[-window_size:]
        ]
        
        if len(recent_scores) < 2:
            return False
        
        score_changes = [
            abs(recent_scores[i] - recent_scores[i-1])
            for i in range(1, len(recent_scores))
        ]
        
        max_change = max(score_changes) if score_changes else 0.0
        converged = max_change < tolerance
        
        logger.info(
            f"Convergence check: window_size={window_size}, "
            f"max_change={max_change:.6f}, tolerance={tolerance:.6f}, "
            f"converged={converged}"
        )
        
        return converged
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику по всем оценкам.
        
        Возвращает:
            Словарь со статистическими показателями
        """
        if not self._evaluation_history:
            return {
                'total_evaluations': 0,
                'statistics': {}
            }
        
        scores_list = self._evaluation_history
        count = len(scores_list)
        
        overall_scores = [s.overall_score for s in scores_list]
        correctness_scores = [s.correctness for s in scores_list]
        completeness_scores = [s.completeness for s in scores_list]
        relevance_scores = [s.relevance for s in scores_list]
        clarity_scores = [s.clarity for s in scores_list]
        
        def compute_stats(values):
            """Вычислить статистику для набора значений."""
            return {
                'mean': statistics.mean(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'median': statistics.median(values)
            }
        
        stats = {
            'total_evaluations': count,
            'overall': compute_stats(overall_scores),
            'correctness': compute_stats(correctness_scores),
            'completeness': compute_stats(completeness_scores),
            'relevance': compute_stats(relevance_scores),
            'clarity': compute_stats(clarity_scores),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return stats
    
    def get_tool_statistics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Получить статистику по инструментам.
        
        Аргументы:
            tool_name: Конкретный инструмент (если None, все)
            
        Возвращает:
            Статистика по инструментам
        """
        if tool_name:
            if tool_name not in self._tool_statistics:
                logger.warning(f"No statistics for tool: {tool_name}")
                return {}
            
            return self._tool_statistics[tool_name]
        
        return self._tool_statistics
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """
        Получить метрики обучения.
        
        Возвращает:
            Метрики обучения (улучшение, сходимость, и т.д.)
        """
        if not self._evaluation_history:
            return {
                'iterations': 0,
                'metrics': {}
            }
        
        first_score = self._evaluation_history[0].overall_score
        last_score = self._evaluation_history[-1].overall_score
        improvement = last_score - first_score
        improvement_percent = (improvement / first_score * 100) if first_score > 0 else 0
        
        # Скорость сходимости (наклон линии тренда)
        if len(self._evaluation_history) > 1:
            scores = [s.overall_score for s in self._evaluation_history]
            convergence_rate = statistics.mean([
                abs(scores[i] - scores[i-1])
                for i in range(1, len(scores))
            ])
        else:
            convergence_rate = 0.0
        
        metrics = {
            'iterations': len(self._evaluation_history),
            'initial_score': first_score,
            'final_score': last_score,
            'total_improvement': improvement,
            'improvement_percent': improvement_percent,
            'convergence_rate': convergence_rate,
            'is_converged': self.check_convergence(),
            'tools_updated_total': len(self._tool_update_history)
        }
        
        return metrics
    
    def compare_systems(
        self,
        other_trainer: 'RLAIFTrainer',
        system_a_name: str = "System A",
        system_b_name: str = "System B"
    ) -> Dict[str, Any]:
        """
        Сравнить результаты обучения двух систем.
        
        Аргументы:
            other_trainer: Другой RLAIF trainer для сравнения
            system_a_name: Название этой системы
            system_b_name: Название другой системы
            
        Возвращает:
            Результаты сравнения
        """
        stats_a = self.get_evaluation_statistics()
        stats_b = other_trainer.get_evaluation_statistics()
        
        metrics_a = self.get_learning_metrics()
        metrics_b = other_trainer.get_learning_metrics()
        
        def calc_improvement_percent(val_a, val_b):
            """Вычислить улучшение в процентах."""
            if val_b == 0 or val_b is None:
                return 0.0
            return ((val_a - val_b) / val_b) * 100
        
        comparison = {
            'system_a': {
                'name': system_a_name,
                'evaluation_count': stats_a.get('total_evaluations', 0),
                'iterations': metrics_a.get('iterations', 0),
                'overall_mean': stats_a.get('overall', {}).get('mean', 0),
                'overall_stdev': stats_a.get('overall', {}).get('stdev', 0),
                'correctness_mean': stats_a.get('correctness', {}).get('mean', 0),
                'completeness_mean': stats_a.get('completeness', {}).get('mean', 0),
                'relevance_mean': stats_a.get('relevance', {}).get('mean', 0),
                'clarity_mean': stats_a.get('clarity', {}).get('mean', 0),
                'improvement_percent': metrics_a.get('improvement_percent', 0),
                'is_converged': metrics_a.get('is_converged', False)
            },
            'system_b': {
                'name': system_b_name,
                'evaluation_count': stats_b.get('total_evaluations', 0),
                'iterations': metrics_b.get('iterations', 0),
                'overall_mean': stats_b.get('overall', {}).get('mean', 0),
                'overall_stdev': stats_b.get('overall', {}).get('stdev', 0),
                'correctness_mean': stats_b.get('correctness', {}).get('mean', 0),
                'completeness_mean': stats_b.get('completeness', {}).get('mean', 0),
                'relevance_mean': stats_b.get('relevance', {}).get('mean', 0),
                'clarity_mean': stats_b.get('clarity', {}).get('mean', 0),
                'improvement_percent': metrics_b.get('improvement_percent', 0),
                'is_converged': metrics_b.get('is_converged', False)
            },
            'comparison': {
                'overall_improvement_percent': calc_improvement_percent(
                    stats_a.get('overall', {}).get('mean', 0),
                    stats_b.get('overall', {}).get('mean', 1)
                ),
                'correctness_improvement_percent': calc_improvement_percent(
                    stats_a.get('correctness', {}).get('mean', 0),
                    stats_b.get('correctness', {}).get('mean', 1)
                ),
                'completeness_improvement_percent': calc_improvement_percent(
                    stats_a.get('completeness', {}).get('mean', 0),
                    stats_b.get('completeness', {}).get('mean', 1)
                ),
                'relevance_improvement_percent': calc_improvement_percent(
                    stats_a.get('relevance', {}).get('mean', 0),
                    stats_b.get('relevance', {}).get('mean', 1)
                ),
                'clarity_improvement_percent': calc_improvement_percent(
                    stats_a.get('clarity', {}).get('mean', 0),
                    stats_b.get('clarity', {}).get('mean', 1)
                ),
                'both_converged': (
                    metrics_a.get('is_converged', False) and
                    metrics_b.get('is_converged', False)
                ),
                'convergence_rate_ratio': (
                    metrics_a.get('convergence_rate', 0) /
                    metrics_b.get('convergence_rate', 1)
                    if metrics_b.get('convergence_rate', 0) > 0 else 0
                )
            }
        }
        
        logger.info(
            f"System comparison: {system_a_name} vs {system_b_name}. "
            f"Overall improvement: "
            f"{comparison['comparison']['overall_improvement_percent']:.2f}%"
        )
        
        return comparison
    
    def save_training_log(self, output_file: str) -> bool:
        """
        Сохранить полный логи обучения в JSON.
        
        Аргументы:
            output_file: Путь для сохранения
            
        Возвращает:
            True если успешно
        """
        try:
            log_data = {
                'metadata': {
                    'saved_at': datetime.utcnow().isoformat(),
                    'config': self._config.to_dict(),
                    'total_evaluations': len(self._evaluation_history),
                    'total_tool_updates': len(self._tool_update_history),
                    'total_iterations': len(self._training_iterations)
                },
                'learning_metrics': self.get_learning_metrics(),
                'evaluation_statistics': self.get_evaluation_statistics(),
                'tool_statistics': self.get_tool_statistics(),
                'evaluations': [
                    {
                        'evaluation_id': score.evaluation_id,
                        'timestamp': score.timestamp,
                        'overall': round(score.overall_score, 6),
                        'correctness': round(score.correctness, 6),
                        'completeness': round(score.completeness, 6),
                        'relevance': round(score.relevance, 6),
                        'clarity': round(score.clarity, 6),
                        'reasoning': score.reasoning,
                        'model': score.model_used
                    }
                    for score in self._evaluation_history
                ],
                'tool_updates': [
                    {
                        'update_id': update.update_id,
                        'tool_name': update.tool_name,
                        'timestamp': update.timestamp,
                        'old_reputation': round(update.old_reputation, 6),
                        'new_reputation': round(update.new_reputation, 6),
                        'delta_reputation': round(update.delta_reputation, 6),
                        'reward': round(update.reward, 6),
                        'learning_rate': update.learning_rate,
                        'success': update.success
                    }
                    for update in self._tool_update_history
                ],
                'training_iterations': self._training_iterations,
                'reputation_timeline': {
                    tool_name: [
                        {'timestamp': ts, 'reputation': round(rep, 6)}
                        for ts, rep in timeline
                    ]
                    for tool_name, timeline in self._reputation_timeline.items()
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Training log saved to {output_file}")
            return True
        
        except Exception as error:
            logger.error(f"Failed to save training log: {error}")
            return False
    
    def save_comparison_report(
        self,
        other_trainer: 'RLAIFTrainer',
        output_file: str,
        system_a_name: str = "System A",
        system_b_name: str = "System B"
    ) -> bool:
        """
        Сохранить отчет сравнения двух систем.
        
        Аргументы:
            other_trainer: Другой trainer для сравнения
            output_file: Путь для сохранения
            system_a_name: Название системы A
            system_b_name: Название системы B
            
        Возвращает:
            True если успешно
        """
        try:
            comparison = self.compare_systems(
                other_trainer,
                system_a_name,
                system_b_name
            )
            
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'systems': comparison,
                'methodology': {
                    'system_a_config': self._config.to_dict(),
                    'system_b_config': other_trainer._config.to_dict(),
                    'metrics': [
                        'overall_score',
                        'correctness',
                        'completeness',
                        'relevance',
                        'clarity'
                    ]
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Comparison report saved to {output_file}")
            return True
        
        except Exception as error:
            logger.error(f"Failed to save comparison report: {error}")
            return False
    
    def get_reputation_dynamics(
        self,
        tool_name: str
    ) -> Dict[str, Any]:
        """
        Получить динамику изменения репутации инструмента.
        
        Аргументы:
            tool_name: Имя инструмента
            
        Возвращает:
            Динамика репутации (timeline, тренд, и т.д.)
        """
        if tool_name not in self._reputation_timeline:
            logger.warning(f"No reputation timeline for tool: {tool_name}")
            return {}
        
        timeline = self._reputation_timeline[tool_name]
        
        if not timeline:
            return {}
        
        timestamps = [t[0] for t in timeline]
        reputations = [t[1] for t in timeline]
        
        changes = [
            reputations[i] - reputations[i-1]
            for i in range(1, len(reputations))
        ]
        
        dynamics = {
            'tool_name': tool_name,
            'initial_reputation': reputations[0],
            'final_reputation': reputations[-1],
            'total_change': reputations[-1] - reputations[0],
            'max_reputation': max(reputations),
            'min_reputation': min(reputations),
            'mean_reputation': statistics.mean(reputations),
            'stdev_reputation': statistics.stdev(reputations) if len(reputations) > 1 else 0.0,
            'total_updates': len(timeline),
            'mean_change_per_update': statistics.mean(changes) if changes else 0.0,
            'max_positive_change': max(changes) if changes else 0.0,
            'max_negative_change': min(changes) if changes else 0.0
        }
        
        return dynamics