"""Оркестратор дообучения инструментов на основе обратной связи"""

from typing import List
from datetime import datetime
import math

from ...entities.base_tool import BaseTool
from ...entities.execution_record import ExecutionRecord
from ..execution.execution_context import ExecutionContext
from ..feedback.feedback_data import FeedbackData
from .training_dataset import TrainingDataset
from .training_dataset_item import TrainingDatasetItem
from .training_statistics import TrainingStatistics


class TrainingOrchestrator:
    """
    Оркестратор дообучения инструментов (пункт 5 из описания системы).
    
    Реализует:
    1. Сбор датасета из выполнений с обратной связью
    2. Обновление параметров инструментов (Policy Gradient)
    3. Обновление эмбеддингов инструментов (Contrastive Learning)
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Инициализация оркестратора.
        
        Args:
            learning_rate: Скорость обучения
        """
        self._dataset = TrainingDataset()
        self._learning_rate = learning_rate
    
    def add_execution_to_dataset(
        self,
        context: ExecutionContext,
        feedbacks: List[FeedbackData]
    ):
        """
        Добавить выполнение в датасет для обучения.
        
        Args:
            context: Контекст выполнения
            feedbacks: Список обратной связи
        """
        item = TrainingDatasetItem(
            task_id=context.task_id,
            task_description=context.task.description if context.task else None,
            task_embedding=context.task.task_embedding if context.task else None,
            domain=context.task.domain if context.task else "general",
            selected_tools=[
                step.selected_tool 
                for step in context.execution_steps 
                if step.selected_tool
            ],
            gradient_traces=context.gradient_traces,
            quality_score=sum(f.quality_score for f in feedbacks) / len(feedbacks) if feedbacks else 0.0,
            execution_time=context.total_time,
            total_cost=context.total_cost,
            feedbacks=feedbacks
        )
        
        self._dataset.add_item(item)
    
    def update_tool(self, tool: BaseTool, item: TrainingDatasetItem):
        """
        Обновить инструмент на основе обратной связи (Policy Gradient).
        
        Формула Policy Gradient: ∇ = (R - baseline) * ∇log P(a|s)
        
        Args:
            tool: Инструмент для обновления
            item: Элемент датасета с обратной связью
        """
        # Находим все случаи, когда этот инструмент был использован
        relevant_traces = [
            gt for gt in item.gradient_traces
            if gt.selected_tool == tool
        ]
        
        if not relevant_traces:
            return
        
        for trace in relevant_traces:
            # Policy Gradient: advantage = reward - baseline
            reward = item.quality_score
            baseline = tool.metadata.reputation
            advantage = reward - baseline
            
            # Обновление репутации с learning rate
            new_reputation = (
                tool.metadata.reputation + 
                self._learning_rate * advantage
            )
            tool.metadata.reputation = max(0.01, min(0.99, new_reputation))
            
            # Обновление статистики
            tool.metadata.quality_scores.append(reward)
            tool.metadata.training_sample_size += 1
            
            # Обновление дисперсии
            if len(tool.metadata.quality_scores) > 1:
                mean = sum(tool.metadata.quality_scores) / len(tool.metadata.quality_scores)
                variance = sum(
                    (s - mean) ** 2 
                    for s in tool.metadata.quality_scores
                ) / len(tool.metadata.quality_scores)
                tool.metadata.variance_estimate = variance
            
            # Обновление времени и стоимости
            total_time = item.execution_time / max(len(item.selected_tools), 1)
            sample_size = tool.metadata.training_sample_size
            tool.metadata.mean_time_answer = (
                (tool.metadata.mean_time_answer * (sample_size - 1) + total_time)
                / sample_size
            )
            
            # Добавление записи в историю
            tool.metadata.execution_history.append(
                ExecutionRecord(
                    task_id=item.task_id,
                    execution_time=item.created_at,
                    time_taken=total_time,
                    cost=tool.mean_cost,
                    quality_score=reward,
                    task_embedding=item.task_embedding,
                    success=reward > 0.5
                )
            )
            
            tool.metadata.last_training_date = datetime.utcnow()
    
    def train_all_tools(self, tools: List[BaseTool]):
        """
        Обучить все инструменты на накопленном датасете.
        
        Args:
            tools: Список инструментов для обучения
        """
        items = self._dataset.get_items()
        
        for tool in tools:
            for item in items:
                self.update_tool(tool, item)
    
    def train_on_successful_executions(
        self,
        tools: List[BaseTool],
        quality_threshold: float = 0.7
    ):
        """
        Обучить инструменты только на успешных выполнениях.
        
        Args:
            tools: Список инструментов
            quality_threshold: Минимальная оценка качества
        """
        items = self._dataset.get_items_by_quality_threshold(quality_threshold)
        
        for tool in tools:
            for item in items:
                self.update_tool(tool, item)
    
    def update_tool_embeddings(
        self,
        tools: List[BaseTool],
        quality_threshold: float = 0.7
    ):
        """
        Обновить эмбеддинги инструментов (Contrastive Learning).
        
        Притягиваем эмбеддинг к успешным задачам,
        отталкиваем от неуспешных.
        
        Args:
            tools: Список инструментов
            quality_threshold: Порог успешности
        """
        successful_items = self._dataset.get_items_by_quality_threshold(
            quality_threshold
        )
        failed_items = [
            item for item in self._dataset.get_items()
            if item.quality_score < quality_threshold
        ]
        
        for tool in tools:
            # Находим задачи, где инструмент был успешен и неуспешен
            successful_tasks = [
                item for item in successful_items
                if tool in item.selected_tools
            ]
            
            failed_tasks = [
                item for item in failed_items
                if tool in item.selected_tools
            ]
            
            if not successful_tasks:
                continue
            
            # Обновляем эмбеддинг
            current_embedding = tool.metadata.capabilities_embedding
            if current_embedding is None:
                continue
            
            new_embedding = current_embedding.copy()
            
            # Притягивание к успешным задачам
            for task in successful_tasks:
                if task.task_embedding is None:
                    continue
                
                for i in range(len(new_embedding)):
                    delta = task.task_embedding[i] - new_embedding[i]
                    new_embedding[i] += (
                        self._learning_rate * delta * task.quality_score
                    )
            
            # Отталкивание от неуспешных задач
            for task in failed_tasks:
                if task.task_embedding is None:
                    continue
                
                for i in range(len(new_embedding)):
                    delta = task.task_embedding[i] - new_embedding[i]
                    new_embedding[i] -= (
                        self._learning_rate * delta * (1 - task.quality_score)
                    )
            
            # Нормализация
            norm = math.sqrt(sum(v * v for v in new_embedding))
            if norm > 0:
                new_embedding = [v / norm for v in new_embedding]
            
            tool.metadata.capabilities_embedding = new_embedding
    
    def get_statistics(self) -> TrainingStatistics:
        """
        Получить статистику обучения.
        
        Returns:
            Статистика обучения
        """
        items = self._dataset.get_items()
        
        if not items:
            return TrainingStatistics()
        
        return TrainingStatistics(
            total_executions=len(items),
            average_quality=sum(i.quality_score for i in items) / len(items),
            success_rate=sum(1 for i in items if i.quality_score >= 0.7) / len(items),
            average_execution_time=sum(i.execution_time for i in items) / len(items),
            average_cost=sum(i.total_cost for i in items) / len(items)
        )
    
    def clear_dataset(self):
        """Очистить датасет"""
        self._dataset.clear()
    
    def get_dataset(self) -> TrainingDataset:
        """Получить датасет для анализа"""
        return self._dataset
