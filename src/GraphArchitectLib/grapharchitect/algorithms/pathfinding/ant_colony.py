"""Муравьиный алгоритм для поиска топ-N путей"""

from typing import List, Dict, Tuple, TypeVar, Generic
import random

from ..graph.base_edge import BaseEdge
from ..graph.graph import GraphW

TEdge = TypeVar('TEdge', bound=BaseEdge)


class AntColonyOptimization(Generic[TEdge]):
    """
    Муравьиный алгоритм (Ant Colony Optimization) для поиска топ-N путей в графе.
    
    Использует феромоны для вероятностного выбора ребер.
    """
    
    def __init__(
        self,
        graph: GraphW[TEdge],
        start: int,
        target: int,
        num_ants: int = 10,
        num_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.5,
        q_value: float = 100.0
    ):
        """
        Инициализация муравьиного алгоритма.
        
        Args:
            graph: Граф
            start: Начальная вершина
            target: Целевая вершина
            num_ants: Количество муравьев
            num_iterations: Количество итераций
            alpha: Вес феромона
            beta: Вес эвристики (1/вес ребра)
            evaporation: Коэффициент испарения (0-1)
            q_value: Константа для обновления феромона
        """
        self._graph = graph
        self._start = start
        self._target = target
        self._num_ants = num_ants
        self._num_iterations = num_iterations
        self._alpha = alpha
        self._beta = beta
        self._evaporation = evaporation
        self._q_value = q_value
        
        # Феромоны на ребрах: {(v, w): pheromone_level}
        self._pheromones: Dict[Tuple[int, int], float] = {}
        
        # Найденные пути
        self._best_paths: List[Tuple[float, List[TEdge]]] = []
        
        # Выполнить поиск
        self._initialize_pheromones()
        self._search()
    
    def _initialize_pheromones(self):
        """Инициализировать феромоны на всех ребрах"""
        for v in range(self._graph.v):
            for edge in self._graph.adj(v):
                self._pheromones[(v, edge.end_v)] = 1.0
    
    def _search(self):
        """Выполнить поиск путей"""
        for iteration in range(self._num_iterations):
            iteration_paths = []
            
            # Каждый муравей строит путь
            for ant in range(self._num_ants):
                path = self._build_path()
                if path:
                    iteration_paths.append(path)
            
            # Обновить феромоны
            self._update_pheromones(iteration_paths)
            
            # Сохранить лучшие пути
            self._best_paths.extend(iteration_paths)
        
        # Сортировать по стоимости
        self._best_paths.sort(key=lambda x: x[0])
    
    def _build_path(self) -> Tuple[float, List[TEdge]]:
        """
        Построить путь одним муравьем.
        
        Returns:
            Кортеж (стоимость, список_ребер) или None если путь не найден
        """
        visited = {self._start}
        path = []
        current = self._start
        total_cost = 0.0
        
        while current != self._target:
            # Получить кандидатов
            candidates = [
                e for e in self._graph.adj(current)
                if e.end_v not in visited
            ]
            
            if not candidates:
                return None  # Тупик
            
            # Выбрать следующее ребро вероятностно
            edge = self._select_next_edge(current, candidates, visited)
            
            path.append(edge)
            total_cost += edge.w
            visited.add(edge.end_v)
            current = edge.end_v
            
            # Защита от слишком длинных путей
            if len(path) > self._graph.v:
                return None
        
        return (total_cost, path)
    
    def _select_next_edge(
        self,
        current: int,
        candidates: List[TEdge],
        visited: set
    ) -> TEdge:
        """
        Выбрать следующее ребро вероятностно.
        
        Вероятность ~ (феромон^alpha) * (эвристика^beta)
        """
        # Вычислить вероятности
        probabilities = []
        total = 0.0
        
        for edge in candidates:
            pheromone = self._pheromones.get((current, edge.end_v), 1.0)
            heuristic = 1.0 / max(edge.w, 0.001)  # 1/вес как эвристика
            
            prob = (pheromone ** self._alpha) * (heuristic ** self._beta)
            probabilities.append(prob)
            total += prob
        
        # Нормализовать
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # Равномерное распределение
            probabilities = [1.0 / len(candidates)] * len(candidates)
        
        # Рулетка
        r = random.random()
        cumulative = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return candidates[i]
        
        return candidates[-1]
    
    def _update_pheromones(self, paths: List[Tuple[float, List[TEdge]]]):
        """Обновить феромоны на основе найденных путей"""
        # Испарение
        for key in self._pheromones:
            self._pheromones[key] *= (1 - self._evaporation)
        
        # Добавление феромона от муравьев
        for cost, path in paths:
            contribution = self._q_value / cost
            
            for edge in path:
                key = (edge.start_v, edge.end_v)
                if key in self._pheromones:
                    self._pheromones[key] += contribution
    
    def get_paths(self, top_n: int) -> List[List[TEdge]]:
        """
        Получить топ-N лучших путей.
        
        Args:
            top_n: Количество путей для возврата
            
        Returns:
            Список путей (без дубликатов)
        """
        # Удалить дубликаты
        unique_paths = {}
        
        for cost, path in self._best_paths:
            # Ключ для идентификации уникальности пути
            path_key = tuple((e.start_v, e.end_v) for e in path)
            
            if path_key not in unique_paths or cost < unique_paths[path_key][0]:
                unique_paths[path_key] = (cost, path)
        
        # Отсортировать и взять топ-N
        sorted_paths = sorted(unique_paths.values(), key=lambda x: x[0])
        return [path for cost, path in sorted_paths[:top_n]]
