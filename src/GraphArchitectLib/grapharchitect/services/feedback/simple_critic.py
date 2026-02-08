"""Простой автоматический критик"""

from abc import ABC, abstractmethod
from typing import List

from .feedback_data import FeedbackData, FeedbackSource
from ..execution.execution_context import ExecutionContext
from ..execution.execution_status import ExecutionStatus
from ...entities.connectors.connector import Connector


class ICriticTool(ABC):
    """Интерфейс критика для автоматической оценки выполнения"""
    
    @abstractmethod
    def evaluate_execution(self, context: ExecutionContext) -> FeedbackData:
        """
        Оценить выполнение задачи.
        
        Args:
            context: Контекст выполнения
            
        Returns:
            Обратная связь
        """
        pass
    
    @abstractmethod
    def check_format_compliance(
        self,
        result: any,
        expected_format: Connector
    ) -> bool:
        """
        Проверить соответствие результата ожидаемому формату.
        
        Args:
            result: Результат выполнения
            expected_format: Ожидаемый формат
            
        Returns:
            True если формат соответствует
        """
        pass
    
    @abstractmethod
    def detect_errors(self, result: any) -> List[str]:
        """
        Обнаружить ошибки в результате.
        
        Args:
            result: Результат выполнения
            
        Returns:
            Список обнаруженных ошибок
        """
        pass


class SimpleCritic(ICriticTool):
    """
    Простая реализация критика на основе эвристик.
    
    В продакшене заменить на ML-модель или LLM.
    """
    
    def evaluate_execution(self, context: ExecutionContext) -> FeedbackData:
        """
        Оценить выполнение задачи.
        
        Оценка основана на:
        1. Успешности всех шагов
        2. Времени выполнения
        3. Наличии результата
        4. Соответствии формату
        """
        feedback = FeedbackData(
            task_id=context.task_id,
            source=FeedbackSource.AUTO_CRITIC,
            success=context.status == ExecutionStatus.COMPLETED
        )
        
        if not feedback.success:
            feedback.quality_score = 0.0
            feedback.comment = "Задача не была выполнена успешно"
            return feedback
        
        # Оценка на основе нескольких факторов
        score = 1.0
        
        # 1. Проверка успешности всех шагов
        if context.execution_steps:
            failed_steps = sum(
                1 for s in context.execution_steps if not s.success
            )
            if failed_steps > 0:
                score *= 0.5
                feedback.detailed_scores["steps_success"] = 0.5
            else:
                feedback.detailed_scores["steps_success"] = 1.0
        
        # 2. Оценка на основе времени выполнения
        if context.execution_steps:
            avg_time = sum(
                s.execution_time for s in context.execution_steps
            ) / len(context.execution_steps)
            
            if avg_time > 10.0:  # Если шаги слишком долгие
                score *= 0.9
                feedback.detailed_scores["time_efficiency"] = 0.9
            else:
                feedback.detailed_scores["time_efficiency"] = 1.0
        
        # 3. Проверка наличия результата
        if context.result is None:
            score *= 0.7
            feedback.detailed_scores["result_presence"] = 0.7
        else:
            feedback.detailed_scores["result_presence"] = 1.0
        
        # 4. Проверка формата выхода
        if context.task and context.task.output_connector:
            format_ok = self.check_format_compliance(
                context.result,
                context.task.output_connector
            )
            if not format_ok:
                score *= 0.8
                feedback.detailed_scores["format_compliance"] = 0.8
            else:
                feedback.detailed_scores["format_compliance"] = 1.0
        
        feedback.quality_score = max(0.0, min(1.0, score))
        feedback.comment = f"Автоматическая оценка: {feedback.quality_score:.2f}"
        
        return feedback
    
    def check_format_compliance(
        self,
        result: any,
        expected_format: Connector
    ) -> bool:
        """
        Проверить соответствие результата ожидаемому формату.
        
        Простая проверка - в реальности нужна более сложная логика.
        """
        if result is None:
            return False
        
        # TODO: Реализовать проверку формата
        return True
    
    def detect_errors(self, result: any) -> List[str]:
        """
        Обнаружить ошибки в результате.
        
        Здесь можно добавить различные проверки специфичные для домена.
        """
        errors = []
        
        if result is None:
            errors.append("Результат равен None")
        
        return errors
