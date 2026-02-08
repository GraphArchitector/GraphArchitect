"""
Конфигурация pytest и общие фикстуры для всех тестов.

Этот файл автоматически загружается pytest перед запуском тестов.
"""

import pytest
import sys
import os
from pathlib import Path

# Добавляем путь к модулю grapharchitect в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== Маркеры тестов ====================

def pytest_configure(config):
    """Регистрация пользовательских маркеров"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "stochastic: marks tests that use randomness (may rarely fail)"
    )


# ==================== Общие фикстуры ====================

@pytest.fixture(scope="session")
def project_root_path():
    """Путь к корню проекта"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def grapharchitect_path(project_root_path):
    """Путь к модулю grapharchitect"""
    return project_root_path / "grapharchitect"


@pytest.fixture
def temp_dir(tmp_path):
    """Временная директория для тестов"""
    return tmp_path


# ==================== Фикстуры для подавления вывода ====================

@pytest.fixture
def suppress_stdout():
    """Подавление stdout во время теста"""
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        yield f


# ==================== Фикстуры для измерения производительности ====================

@pytest.fixture
def timer():
    """Таймер для измерения времени выполнения"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            if self.start_time:
                self.elapsed = time.time() - self.start_time
                return self.elapsed
            return None
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, *args):
            self.stop()
    
    return Timer()


# ==================== Автоматическое логирование ====================

@pytest.fixture(autouse=True)
def log_test_info(request):
    """Автоматическое логирование информации о тесте"""
    test_name = request.node.name
    print(f"\n▶ Running test: {test_name}")
    
    yield
    
    # После теста
    if hasattr(request.node, 'rep_call') and request.node.rep_call.failed:
        print(f"✗ Test FAILED: {test_name}")
    else:
        print(f"✓ Test PASSED: {test_name}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Сохранение результата теста для использования в фикстурах"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# ==================== Настройки для случайных тестов ====================

@pytest.fixture
def seed_random():
    """Фиксация seed для воспроизводимости случайных тестов"""
    import random
    random.seed(42)
    yield
    # Восстановление случайности после теста
    random.seed()


# ==================== Очистка после тестов ====================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Автоматическая очистка после каждого теста"""
    yield
    
    # Здесь можно добавить очистку глобальных состояний
    # Например, очистка кешей, закрытие соединений и т.д.


# ==================== Фикстуры для моков ====================

@pytest.fixture
def mock_embedding_service():
    """Моковый сервис векторизации"""
    from grapharchitect.services.embedding.simple_embedding_service import SimpleEmbeddingService
    return SimpleEmbeddingService(dimension=64)


@pytest.fixture
def mock_selector():
    """Моковый селектор инструментов"""
    from grapharchitect.services.selection.instrument_selector import InstrumentSelector
    return InstrumentSelector(temperature_constant=1.0)


# ==================== Настройки pytest ====================

def pytest_collection_modifyitems(config, items):
    """Модификация собранных тестов"""
    # Автоматически добавляем маркер "unit" к тестам без других маркеров
    for item in items:
        if not any(mark.name in ["integration", "slow", "stochastic"] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# ==================== Параметризация для разных конфигураций ====================

@pytest.fixture(params=[0.1, 1.0, 10.0])
def temperature_constant(request):
    """Параметризованная константа температуры"""
    return request.param


@pytest.fixture(params=[1, 3, 5])
def top_k_values(request):
    """Параметризованные значения top-k"""
    return request.param


# ==================== Вывод информации о тестовой сессии ====================

def pytest_sessionstart(session):
    """Вызывается при старте тестовой сессии"""
    print("\n" + "="*70)
    print("🧪 ЗАПУСК ТЕСТОВ GRAPHARCHITECT")
    print("="*70)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Pytest version: {pytest.__version__}")
    print(f"Project root: {project_root}")
    print("="*70 + "\n")


def pytest_sessionfinish(session, exitstatus):
    """Вызывается при завершении тестовой сессии"""
    print("\n" + "="*70)
    print("🏁 ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*70)
    
    if exitstatus == 0:
        print("✓ Все тесты прошли успешно!")
    else:
        print(f"✗ Некоторые тесты провалились (exit code: {exitstatus})")
    
    print("="*70 + "\n")


# ==================== Пропуск тестов при определенных условиях ====================

def pytest_runtest_setup(item):
    """Проверка условий перед запуском теста"""
    # Пример: пропустить медленные тесты если указан --fast
    if "slow" in item.keywords and item.config.getoption("--fast", default=False):
        pytest.skip("skipping slow test in fast mode")


def pytest_addoption(parser):
    """Добавление пользовательских опций командной строки"""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="run only fast tests"
    )
    parser.addoption(
        "--integration-only",
        action="store_true",
        default=False,
        help="run only integration tests"
    )


# ==================== Отчет о покрытии ====================

@pytest.fixture(scope="session", autouse=True)
def coverage_report(request):
    """Автоматический отчет о покрытии после всех тестов"""
    yield
    
    # После всех тестов
    if request.config.getoption("--cov", default=None):
        print("\n" + "="*70)
        print("📊 ОТЧЕТ О ПОКРЫТИИ КОДА")
        print("="*70)
        print("Детальный отчет будет сохранен в htmlcov/index.html")
        print("="*70 + "\n")
