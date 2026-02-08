"""
Тесты для алгоритмов поиска путей в графе.

Тестируются:
- Dijkstra - алгоритм Дейкстры
- AStar - A* с эвристикой
- YenKShortestPaths - топ-K путей
- AntColonyOptimization - муравьиный алгоритм
"""

import pytest
import math
from grapharchitect.algorithms.graph.graph import GraphW
from grapharchitect.algorithms.graph.base_edge import BaseEdge
from grapharchitect.algorithms.pathfinding.dijkstra import Dijkstra
from grapharchitect.algorithms.pathfinding.astar import AStar
from grapharchitect.algorithms.pathfinding.yen import YenKShortestPaths
from grapharchitect.algorithms.pathfinding.ant_colony import AntColonyOptimization


# ==================== Фикстуры ====================

@pytest.fixture
def simple_graph():
    """
    Простой граф для тестирования.
    
    Структура:
        0 --7-> 1 --15-> 3 --6-> 4
        |       |        |       |
        9      10       11       9
        |       |        |       |
        v       v        v       v
        2 --2-> 5 <------+-------+
        |
       14
        |
        v
        5
    """
    graph = GraphW[BaseEdge](6)
    graph.add_edge(BaseEdge(0, 1, 7))
    graph.add_edge(BaseEdge(0, 2, 9))
    graph.add_edge(BaseEdge(0, 5, 14))
    graph.add_edge(BaseEdge(1, 2, 10))
    graph.add_edge(BaseEdge(1, 3, 15))
    graph.add_edge(BaseEdge(2, 3, 11))
    graph.add_edge(BaseEdge(2, 5, 2))
    graph.add_edge(BaseEdge(3, 4, 6))
    graph.add_edge(BaseEdge(4, 5, 9))
    return graph


@pytest.fixture
def coordinates():
    """Координаты вершин для эвристики"""
    return {
        0: (0, 0),
        1: (1, 4),
        2: (1, 0),
        3: (2, 5),
        4: (3, 5),
        5: (2, 0)
    }


@pytest.fixture
def disconnected_graph():
    """Граф с несвязными компонентами"""
    graph = GraphW[BaseEdge](6)
    # Компонента 1: 0-1-2
    graph.add_edge(BaseEdge(0, 1, 5))
    graph.add_edge(BaseEdge(1, 2, 3))
    # Компонента 2: 3-4-5
    graph.add_edge(BaseEdge(3, 4, 2))
    graph.add_edge(BaseEdge(4, 5, 4))
    return graph


# ==================== Тесты Dijkstra ====================

class TestDijkstra:
    """Тесты алгоритма Дейкстры"""
    
    def test_distances_from_start(self, simple_graph):
        """Проверка правильности расстояний от начальной вершины"""
        dijkstra = Dijkstra(simple_graph, 0)
        expected_distances = [0, 7, 9, 20, 26, 11]
        assert dijkstra.distances == expected_distances
    
    def test_shortest_path_to_vertex_4(self, simple_graph):
        """Проверка кратчайшего пути до вершины 4"""
        dijkstra = Dijkstra(simple_graph, 0)
        assert dijkstra.has_path_to(4)
        assert dijkstra.distance_to(4) == 26
    
    def test_shortest_path_to_vertex_3(self, simple_graph):
        """Проверка кратчайшего пути до вершины 3"""
        dijkstra = Dijkstra(simple_graph, 0)
        assert dijkstra.has_path_to(3)
        assert dijkstra.distance_to(3) == 20
    
    def test_path_to_self(self, simple_graph):
        """Путь до самой себя должен быть 0"""
        dijkstra = Dijkstra(simple_graph, 0)
        assert dijkstra.distance_to(0) == 0
    
    def test_no_path_in_disconnected_graph(self, disconnected_graph):
        """Отсутствие пути в несвязном графе"""
        dijkstra = Dijkstra(disconnected_graph, 0)
        assert not dijkstra.has_path_to(5)
        assert dijkstra.distance_to(5) == float('inf')
    
    def test_path_within_component(self, disconnected_graph):
        """Путь внутри связной компоненты"""
        dijkstra = Dijkstra(disconnected_graph, 0)
        assert dijkstra.has_path_to(2)
        assert dijkstra.distance_to(2) == 8  # 0->1->2: 5+3


# ==================== Тесты A* ====================

class TestAStar:
    """Тесты алгоритма A*"""
    
    def test_without_heuristic_equals_dijkstra(self, simple_graph):
        """A* без эвристики эквивалентен Дейкстре"""
        astar = AStar(simple_graph, 0, 4, None)
        dijkstra = Dijkstra(simple_graph, 0)
        
        assert astar.distance_to(4) == dijkstra.distance_to(4)
    
    def test_with_zero_heuristic(self, simple_graph):
        """A* с нулевой эвристикой"""
        def zero_heuristic(v: int, target: int) -> float:
            return 0.0
        
        astar = AStar(simple_graph, 0, 4, zero_heuristic)
        assert astar.distance_to(4) == 26
    
    def test_with_euclidean_heuristic(self, simple_graph, coordinates):
        """A* с евклидовой эвристикой"""
        def euclidean_heuristic(v: int, target: int) -> float:
            x1, y1 = coordinates[v]
            x2, y2 = coordinates[target]
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        astar = AStar(simple_graph, 0, 5, euclidean_heuristic)
        assert astar.has_path_to(5)
        assert astar.distance_to(5) == 11  # 0->2->5
    
    def test_path_reconstruction(self, simple_graph):
        """Восстановление пути"""
        astar = AStar(simple_graph, 0, 4, None)
        path = astar.get_path()
        
        assert path is not None
        assert len(path) > 0
        assert path[0].start_v == 0
        assert path[-1].end_v == 4
    
    def test_no_path_to_target(self, disconnected_graph):
        """A* когда пути до цели нет"""
        astar = AStar(disconnected_graph, 0, 5, None)
        assert not astar.has_path_to(5)
        assert astar.get_path() is None


# ==================== Тесты Yen ====================

class TestYen:
    """Тесты алгоритма Йена для топ-K путей"""
    
    def test_single_shortest_path(self, simple_graph):
        """Поиск одного кратчайшего пути"""
        yen = YenKShortestPaths(simple_graph, 0, 4)
        paths = yen.get_paths(1)
        
        assert len(paths) == 1
        assert len(paths[0]) > 0
    
    def test_multiple_paths(self, simple_graph):
        """Поиск нескольких путей"""
        yen = YenKShortestPaths(simple_graph, 0, 4)
        paths = yen.get_paths(3)
        
        # Должно быть найдено несколько путей
        assert len(paths) >= 1
        assert len(paths) <= 3
        
        # Первый путь должен быть кратчайшим
        if len(paths) > 1:
            cost1 = sum(e.w for e in paths[0])
            cost2 = sum(e.w for e in paths[1])
            assert cost1 <= cost2
    
    def test_no_path_returns_empty(self, disconnected_graph):
        """Отсутствие пути возвращает пустой список"""
        yen = YenKShortestPaths(disconnected_graph, 0, 5)
        paths = yen.get_paths(3)
        
        assert paths == []
    
    def test_paths_are_different(self, simple_graph):
        """Все найденные пути должны быть различными"""
        yen = YenKShortestPaths(simple_graph, 0, 5)
        paths = yen.get_paths(5)
        
        # Сравниваем пути попарно
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                # Пути должны различаться
                path_i = [(e.start_v, e.end_v) for e in paths[i]]
                path_j = [(e.start_v, e.end_v) for e in paths[j]]
                assert path_i != path_j


# ==================== Тесты ACO ====================

class TestAntColonyOptimization:
    """Тесты муравьиного алгоритма"""
    
    def test_finds_path(self, simple_graph):
        """ACO находит хотя бы один путь"""
        aco = AntColonyOptimization(
            simple_graph, 0, 4,
            num_ants=5,
            num_iterations=10
        )
        paths = aco.get_paths(1)
        
        assert len(paths) >= 1
        assert len(paths[0]) > 0
    
    def test_multiple_paths(self, simple_graph):
        """ACO находит несколько путей"""
        aco = AntColonyOptimization(
            simple_graph, 0, 5,
            num_ants=10,
            num_iterations=20
        )
        paths = aco.get_paths(3)
        
        assert len(paths) >= 1
    
    def test_paths_reach_target(self, simple_graph):
        """Все пути достигают целевой вершины"""
        aco = AntColonyOptimization(
            simple_graph, 0, 4,
            num_ants=5,
            num_iterations=10
        )
        paths = aco.get_paths(5)
        
        for path in paths:
            assert path[-1].end_v == 4
    
    def test_with_custom_parameters(self, simple_graph):
        """ACO с пользовательскими параметрами"""
        aco = AntColonyOptimization(
            simple_graph, 0, 5,
            num_ants=20,
            num_iterations=50,
            alpha=2.0,  # Больший вес феромона
            beta=3.0,   # Больший вес эвристики
            evaporation=0.3,
            q_value=200.0
        )
        paths = aco.get_paths(2)
        
        assert len(paths) >= 1
    
    def test_no_path_returns_empty(self, disconnected_graph):
        """ACO возвращает пустой список если пути нет"""
        aco = AntColonyOptimization(
            disconnected_graph, 0, 5,
            num_ants=5,
            num_iterations=10
        )
        paths = aco.get_paths(3)
        
        assert len(paths) == 0


# ==================== Тесты графа ====================

class TestGraphW:
    """Тесты базовой структуры графа"""
    
    def test_graph_creation(self):
        """Создание графа"""
        graph = GraphW[BaseEdge](5)
        assert graph.v == 5
        assert graph.e == 0
    
    def test_add_edge(self):
        """Добавление ребра"""
        graph = GraphW[BaseEdge](3)
        edge = BaseEdge(0, 1, 5.0)
        graph.add_edge(edge)
        
        assert graph.e == 1
        assert edge in graph.adj(0)
    
    def test_adjacency_list(self):
        """Список смежности"""
        graph = GraphW[BaseEdge](3)
        graph.add_edge(BaseEdge(0, 1, 1.0))
        graph.add_edge(BaseEdge(0, 2, 2.0))
        
        adj = graph.adj(0)
        assert len(adj) == 2
    
    def test_remove_edge(self):
        """Удаление ребра"""
        graph = GraphW[BaseEdge](3)
        edge = BaseEdge(0, 1, 5.0)
        graph.add_edge(edge)
        assert graph.e == 1
        
        graph.remove_edge(edge)
        assert graph.e == 0
    
    def test_all_edges(self):
        """Получение всех ребер"""
        graph = GraphW[BaseEdge](3)
        graph.add_edge(BaseEdge(0, 1, 1.0))
        graph.add_edge(BaseEdge(1, 2, 2.0))
        
        edges = graph.edges()
        assert len(edges) == 2


# ==================== Тесты BaseEdge ====================

class TestBaseEdge:
    """Тесты базового ребра"""
    
    def test_edge_creation(self):
        """Создание ребра"""
        edge = BaseEdge(0, 1, 5.0)
        assert edge.start_v == 0
        assert edge.end_v == 1
        assert edge.w == 5.0
    
    def test_either_method(self):
        """Метод either возвращает начальную вершину"""
        edge = BaseEdge(3, 7, 2.0)
        assert edge.either() == 3
    
    def test_other_method(self):
        """Метод other возвращает другую вершину"""
        edge = BaseEdge(3, 7, 2.0)
        assert edge.other(3) == 7
        assert edge.other(7) == 3
    
    def test_other_method_invalid_vertex(self):
        """Метод other с неверной вершиной выбрасывает ошибку"""
        edge = BaseEdge(3, 7, 2.0)
        with pytest.raises(ValueError):
            edge.other(5)
    
    def test_edge_equality(self):
        """Равенство ребер"""
        edge1 = BaseEdge(0, 1, 5.0)
        edge2 = BaseEdge(0, 1, 5.0)
        edge3 = BaseEdge(0, 2, 5.0)
        
        assert edge1 == edge2
        assert edge1 != edge3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
