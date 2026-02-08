"""
Конфигурация pytest для Web тестов.
"""

import pytest
import sys
from pathlib import Path

# Добавляем пути к модулям
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))


# ==================== Фикстуры ====================

@pytest.fixture(scope="session")
def repository():
    """Repository для тестов"""
    from repository import get_repository
    return get_repository()


@pytest.fixture(scope="session")
def grapharchitect_bridge():
    """GraphArchitect Bridge для тестов"""
    try:
        from grapharchitect_bridge import get_bridge
        return get_bridge()
    except Exception as e:
        pytest.skip(f"GraphArchitect не доступен: {e}")


@pytest.fixture
def sample_agent(repository):
    """Тестовый агент"""
    return repository.get_agent("agent-classifier-gpt4")


@pytest.fixture
def sample_agents_list(repository):
    """Список тестовых агентов"""
    return [
        repository.get_agent("agent-classifier-gpt4"),
        repository.get_agent("agent-classifier-claude"),
        repository.get_agent("agent-classifier-local")
    ]


# ==================== Настройки pytest ====================

def pytest_configure(config):
    """Настройка pytest"""
    config.addinivalue_line(
        "markers", "integration: integration tests with GraphArchitect"
    )
    config.addinivalue_line(
        "markers", "async_test: async tests requiring event loop"
    )


# ==================== Логирование ====================

@pytest.fixture(autouse=True)
def log_test_info(request):
    """Логирование информации о тестах"""
    test_name = request.node.name
    print(f"\n▶ {test_name}")
    
    yield
    
    if hasattr(request.node, 'rep_call') and request.node.rep_call.failed:
        print(f"[FAILED]")
    else:
        print(f"[PASSED]")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Сохранение результата теста"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
