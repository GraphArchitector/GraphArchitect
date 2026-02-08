"""Перечисление алгоритмов поиска пути"""

from enum import Enum


class PathfindingAlgorithm(Enum):
    """Доступные алгоритмы поиска пути в графе"""
    
    DIJKSTRA = "dijkstra"      # Алгоритм Дейкстры (1 путь)
    ASTAR = "astar"            # A* с эвристикой (1 путь)
    YEN = "yen"                # Алгоритм Йена (топ-K путей)
    ANT_COLONY = "ant_colony"  # Муравьиный алгоритм (топ-K путей)
