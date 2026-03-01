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
        "raw_text": []
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
                
                report['raw_text'].append(content)
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
print(f"Текст ответа:\n")
for text in review['raw_text']:
    print(text)

```

### Проверки безопасности

```python
message = f"""
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
message = f"""
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
- Процесс проверки написанного кода, поиск уязвимостей

---

**Следующий workflow**: [Research Workflow](research_workflow.md)
