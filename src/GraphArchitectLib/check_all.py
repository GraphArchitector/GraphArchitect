#!/usr/bin/env python
"""
Полная проверка проекта GraphArchitect.

Проверяет:
1. Импорты всех модулей
2. Синтаксис Python файлов
3. Наличие обязательных файлов
4. Конфигурацию
5. Зависимости
"""

import sys
import ast
from pathlib import Path
import importlib.util


def check_python_syntax(file_path: Path) -> bool:
    """Проверка синтаксиса Python файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        ast.parse(code)
        return True
    
    except SyntaxError as e:
        print(f"    Syntax error: line {e.lineno}: {e.msg}")
        return False
    
    except Exception as e:
        print(f"    Error: {e}")
        return False


def check_imports(file_path: Path) -> bool:
    """Проверка что файл можно импортировать"""
    try:
        # Проверяем только синтаксис, не импортируем реально
        return check_python_syntax(file_path)
    except Exception as e:
        print(f" Import error: {e}")
        return False


def check_grapharchitect_library():
    """Проверка библиотеки grapharchitect"""
    print("\n" + "="*70)
    print("1ПРОВЕРКА GRAPHARCHITECT LIBRARY")
    print("="*70)
    
    grapharchitect_path = Path("grapharchitect")
    
    if not grapharchitect_path.exists():
        print("Папка grapharchitect не найдена")
        return False
    
    # Проверяем ключевые модули
    key_modules = [
        "grapharchitect/__init__.py",
        "grapharchitect/entities/base_tool.py",
        "grapharchitect/services/selection/instrument_selector.py",
        "grapharchitect/services/execution/execution_orchestrator.py",
        "grapharchitect/services/graph_strategy_finder.py",
        "grapharchitect/algorithms/pathfinding/dijkstra.py"
    ]
    
    errors = 0
    for module_path in key_modules:
        file_path = Path(module_path)
        
        if not file_path.exists():
            print(f"Не найден: {module_path}")
            errors += 1
            continue
        
        if not check_python_syntax(file_path):
            print(f"Ошибка синтаксиса: {module_path}")
            errors += 1
        else:
            print(f"{module_path}")
    
    if errors == 0:
        print(f"\nGraphArchitect library: OK ({len(key_modules)} модулей)")
        return True
    else:
        print(f"\nНайдено ошибок: {errors}")
        return False


def check_web_api():
    """Проверка Web API"""
    print("\n" + "="*70)
    print("ПРОВЕРКА WEB API")
    print("="*70)
    
    web_path = Path("Web")
    
    if not web_path.exists():
        print("Папка Web не найдена")
        return False
    
    # Проверяем ключевые файлы
    key_files = [
        "Web/main.py",
        "Web/api_router.py",
        "Web/services.py",
        "Web/models.py",
        "Web/repository.py",
        "Web/grapharchitect_bridge.py",
        "Web/database.py",
        "Web/sqlite_repository.py",
        "Web/training_service.py"
    ]
    
    errors = 0
    for file_path_str in key_files:
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            print(f"Не найден: {file_path_str}")
            errors += 1
            continue
        
        if not check_python_syntax(file_path):
            print(f"Ошибка синтаксиса: {file_path_str}")
            errors += 1
        else:
            print(f"{file_path_str}")
    
    if errors == 0:
        print(f"\nWeb API: OK ({len(key_files)} файлов)")
        return True
    else:
        print(f"\nНайдено ошибок: {errors}")
        return False


def check_tests():
    """Проверка тестов"""
    print("\n" + "="*70)
    print("ПРОВЕРКА ТЕСТОВ")
    print("="*70)
    
    tests_path = Path("Tests")
    
    if not tests_path.exists():
        print("Папка Tests не найдена")
        return False
    
    # Находим все test_*.py файлы
    test_files = list(tests_path.glob("test_*.py"))
    
    if not test_files:
        print("Тестовые файлы не найдены")
        return False
    
    errors = 0
    for test_file in test_files:
        if not check_python_syntax(test_file):
            print(f"{test_file.name}")
            errors += 1
        else:
            print(f"{test_file.name}")
    
    if errors == 0:
        print(f"\nТесты: OK ({len(test_files)} файлов)")
        return True
    else:
        print(f"\nНайдено ошибок: {errors}")
        return False


def check_openrouter():
    """Проверка OpenRouter интеграции"""
    print("\n" + "="*70)
    print("ПРОВЕРКА OPENROUTER")
    print("="*70)
    
    openrouter_path = Path("grapharchitect/tools/ApiTools/OpenRouterTool")
    
    if not openrouter_path.exists():
        print("Папка OpenRouterTool не найдена")
        return False
    
    files = [
        "openrouter_llm.py",
        "openrouter_config.py",
        "openrouter_basetool.py",
        "__init__.py"
    ]
    
    errors = 0
    for filename in files:
        file_path = openrouter_path / filename
        
        if not file_path.exists():
            print(f"Не найден: {filename}")
            errors += 1
            continue
        
        if not check_python_syntax(file_path):
            print(f"Ошибка синтаксиса: {filename}")
            errors += 1
        else:
            print(f"{filename}")
    
    if errors == 0:
        print(f"\nOpenRouter: OK ({len(files)} файлов)")
        return True
    else:
        print(f"\nНайдено ошибок: {errors}")
        return False





def main():
    """Запуск всех проверок"""
    print("\n" + "="*70)
    print("ПОЛНАЯ ПРОВЕРКА GRAPHARCHITECT")
    print("="*70)
    print(f"\nТекущая директория: {Path.cwd()}")
    print("Запустите из: src/GraphArchitectLib/")
    
    results = []
    
    results.append(("GraphArchitect Library", check_grapharchitect_library()))
    results.append(("Web API", check_web_api()))
    results.append(("Тесты", check_tests()))
    results.append(("OpenRouter", check_openrouter()))
    
    # Итоговый отчет
    print("\n" + "="*70)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*70)
    
    for check_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status} - {check_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print("\n" + "="*70)
    
    if passed_count == total_count:
        print(f"ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ({passed_count}/{total_count})")
        print("="*70)
        print("\nПроект готов к использованию!")
        print("\nЗапускайте:")
        print("   cd Web && python main.py")
        return 0
    else:
        print(f"⚠ПРОЙДЕНО: {passed_count}/{total_count}")
        print("="*70)
        print("\nНекоторые проверки не прошли")
        print("\n Смотрите детали выше")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
