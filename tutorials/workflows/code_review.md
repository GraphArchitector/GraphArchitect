# Workflow: Автоматическое ревью кода

**Сценарий**: Code review с использованием GraphArchitect  
**Сложность**: Продвинутая  
**Время**: 30 минут

---

## Описание задачи

Автоматическое ревью кода включает:
1. Парсинг и понимание кода
2. Проверка стиля и конвенций
3. Анализ потенциальных багов
4. Проверка безопасности
5. Предложение улучшений

---

## Архитектура Workflow

```
Исходный код
    ↓
┌─────────────────────────────────┐
│ Шаг 1: Code Parsing             │
│ - Синтаксический анализ         │
│ - Извлечение структуры          │
│ - Определение языка             │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 2: Style Check              │
│ - PEP8/ESLint проверка          │
│ - Naming conventions            │
│ - Code formatting               │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 3: Bug Detection            │
│ - Логические ошибки             │
│ - Потенциальные exception       │
│ - Edge cases                    │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 4: Security Analysis        │
│ - SQL injection                 │
│ - XSS vulnerabilities           │
│ - Hardcoded secrets             │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Шаг 5: Improvement Suggestions  │
│ - Рефакторинг                   │
│ - Оптимизация                   │
│ - Best practices                │
└────────────┬────────────────────┘
             ↓
Code Review Report
```

---

## Реализация

### Базовый code review

```python
import requests

def review_code(code, language="python"):
    """
    Автоматическое ревью кода.
    
    Args:
        code: Исходный код для ревью
        language: Язык программирования
        
    Returns:
        Отчет о ревью
    """
    message = f"""
    Провести code review для следующего {language} кода:
    
    ```{language}
    {code}
    ```
    
    Проверить:
    1. Стиль и конвенции
    2. Потенциальные баги
    3. Безопасность
    4. Возможности улучшения
    """
    
    response = requests.post(
        "http://localhost:8000/api/chat/codereview/message/stream",
        data={
            "message": message,
            "planning_algorithm": "yen_5"
        },
        stream=True
    )
    
    report = {
        "issues": [],
        "suggestions": [],
        "tools_used": []
    }
    
    for line in response.iter_lines():
        if line:
            import json
            chunk = json.loads(line)
            
            if chunk['type'] == 'agent_selected':
                report['tools_used'].append(chunk['agent_id'])
            
            elif chunk['type'] == 'text':
                # Парсинг результата
                content = chunk['content']
                
                if "Issue:" in content:
                    report['issues'].append(content)
                elif "Suggestion:" in content:
                    report['suggestions'].append(content)
    
    return report


# Пример использования
code = '''
def process_user_data(username, password):
    query = f"SELECT * FROM users WHERE username='{username}'"
    cursor.execute(query)
    user = cursor.fetchone()
    
    if user[1] == password:
        return user
    return None
'''

review = review_code(code)

print(f"Найдено проблем: {len(review['issues'])}")
print(f"Предложений: {len(review['suggestions'])}")
print(f"Использовано инструментов: {len(review['tools_used'])}")
```

---

## Интеграция с Git

### Pre-commit hook

```python
#!/usr/bin/env python
"""
Pre-commit hook для автоматического ревью.
.git/hooks/pre-commit
"""

import subprocess
import requests

def get_changed_files():
    """Получить измененные файлы"""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only'],
        capture_output=True,
        text=True
    )
    return result.stdout.strip().split('\n')

def review_file(filepath):
    """Ревью одного файла"""
    with open(filepath, 'r') as f:
        code = f.read()
    
    # Отправка на ревью
    response = requests.post(
        "http://localhost:8000/api/chat/codereview/message",
        data={
            "message": f"Review code in {filepath}:\n{code}"
        }
    )
    
    return response.json()

# Проверка всех измененных файлов
files = [f for f in get_changed_files() if f.endswith('.py')]

issues_found = False
for filepath in files:
    review = review_file(filepath)
    
    if review.get('issues'):
        print(f"Issues in {filepath}:")
        for issue in review['issues']:
            print(f"  - {issue}")
        issues_found = True

if issues_found:
    print("\nCode review found issues. Fix them before committing.")
    exit(1)
else:
    print("Code review passed!")
    exit(0)
```

### CI/CD Pipeline

```yaml
# .github/workflows/code-review.yml
name: Automated Code Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup GraphArchitect
        run: |
          docker run -d -p 8000:8000 grapharchitect:latest
          
      - name: Review changed files
        run: |
          python scripts/automated_review.py
          
      - name: Comment on PR
        uses: actions/github-script@v5
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: process.env.REVIEW_REPORT
            })
```

---

## Специализированные проверки

### Безопасность (Security)

```python
message = """
Проверить код на уязвимости:
- SQL injection
- XSS
- CSRF
- Hardcoded secrets
- Insecure dependencies

Код:
{code}
"""
```

**Инструменты**:
- Style Checker → находит паттерны уязвимостей
- Strict QA → валидация безопасности

### Производительность (Performance)

```python
message = """
Анализ производительности кода:
- Сложность алгоритмов (Big O)
- Узкие места
- Возможности оптимизации

Код:
{code}
"""
```

**Инструменты**:
- Trend Analyzer → поиск паттернов
- Technical Writer → рекомендации

---

## Метрики качества ревью

### KPI

| Метрика | Цель | Измерение |
|---------|------|-----------|
| Обнаружение багов | > 85% | % найденных vs всех багов |
| False positives | < 15% | % ложных срабатываний |
| Полнота проверки | > 90% | % покрытия правил |
| Время ревью | < 30s | На 100 строк кода |
| Качество рекомендаций | > 80% | % применимых советов |

### Сравнение с человеком

```
Человек-ревьюер:
  Время: 10-20 минут на файл
  Качество: 95%
  Стоимость: $50-100/час
  
GraphArchitect:
  Время: 10-30 секунд на файл
  Качество: 75-85%
  Стоимость: $0.05-0.20 на файл
  
Гибрид (рекомендуется):
  GraphArchitect фильтрует → Человек проверяет важное
  Время: 2-5 минут
  Качество: 95%+
  Стоимость: $10-20/час
```

---

## Расширенные сценарии

### Автоматическое исправление

```python
def review_and_fix(code):
    """Ревью + автоматическое исправление простых проблем"""
    
    # Шаг 1: Ревью
    review_result = review_code(code)
    
    # Шаг 2: Если есть простые проблемы (стиль)
    if has_style_issues(review_result):
        # Автофикс через Style Improver
        fixed_code = improve_code_style(code)
        return fixed_code
    
    return code
```

### Генерация тестов

```python
message = """
Сгенерировать unit тесты для кода:

{code}

Требования:
- Покрытие всех функций
- Edge cases
- Pytest framework
"""
```

---

## Итоги

### Вы создали

- Workflow для автоматического code review
- 5-шаговый процесс проверки
- Интеграцию с Git (pre-commit hooks)
- CI/CD pipeline для PR

### Применение

- Pre-commit hooks
- CI/CD pipelines
- Automated PR reviews
- Continuous code quality monitoring
- Security audits

### Ограничения

GraphArchitect не заменяет человека, но:
- Находит 75-85% проблем автоматически
- Экономит время ревьюера
- Обеспечивает consistency
- Обучается на вашей кодовой базе

---

**Следующий workflow**: [Research Workflow](research_workflow.md)
