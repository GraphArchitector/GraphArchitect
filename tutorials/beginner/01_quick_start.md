# Туториал 1: Быстрый старт

**Уровень**: Начинающий  
**Время**: 5 минут  
**Цель**: Запустить GraphArchitect и выполнить первую задачу

---

## Что вы узнаете

- Как установить и настроить GraphArchitect
- Как запустить Web API сервер
- Как отправить первый запрос
- Как интерпретировать результаты

---

## Предварительные требования

- Python 3.8+ установлен
- Базовое понимание командной строки
- 5 минут

---

## Шаг 1: Установка зависимостей

```bash
cd .\src\GraphArchitectLib\Web

# Установить зависимости
pip install -r requirements.txt
```

**Что происходит**: Устанавливаются FastAPI, SQLite драйверы и другие библиотеки.

**Время**: 1-2 минуты

---

## Шаг 2: Инициализация базы данных

```bash
# Создать таблицы БД
python db_manager.py init

# Загрузить инструменты в БД
python db_manager.py load_agents
```

**Что происходит**: 
- Создается SQLite база данных `grapharchitect.db`
- Загружается 27инструментов (agents) в таблицу

**Ожидаемый вывод**:
```
DATABASE INITIALIZATION
======================================================================

[OK] Database initialized: grapharchitect.db

Tables:
  [OK] agents               (27 records)
  [OK] chats                (0 records)
  [OK] documents            (0 records)
  [OK] executions           (0 records)
  [OK] feedbacks            (0 records)
  [OK] sqlite_sequence      (0 records)
  [OK] tool_metrics         (0 records)
  [OK] workflows            (0 records)
```

---

## Шаг 3: Запуск сервера

```bash
python main.py
```

**Что происходит**: Запускается FastAPI сервер с GraphArchitect.

**Ожидаемый вывод**:
```
INFO - Initializing SQLite database: grapharchitect.db
INFO - Database tables created/verified
INFO - Using SQLite repository
INFO - GraphArchitect integration activated
INFO - Initializing GraphArchitectBridge...
INFO - GraphArchitectBridge ready (27 tools)
INFO - Starting server on port 8000
INFO - Web interface: http://127.0.0.1:8000
```

**Важно**: Оставьте терминал открытым, сервер работает.

---

## Шаг 4: Открытие Web интерфейса

Откройте в браузере:
```
http://localhost:8000
```

**Что вы увидите**:
- Интерфейс чата
- Библиотека инструментов справа, в панели
- Выбор алгоритма планирования справа вверху

---

## Шаг 5: Отправка первого запроса

В поле ввода внизу страницы напишите:

```
Классифицировать этот текст: "Отличный продукт, рекомендую!"
```

Нажмите кнопку отправки или Enter.

**Что происходит**:
1. GraphArchitect парсит запрос через NLI
2. Определяет нужные коннекторы: `text|question` → `text|category`
3. Ищет путь в графе инструментов (алгоритм Yen, топ-5 путей)
4. Выбирает лучший инструмент через softmax
5. Выполняет задачу
6. Обучается на результате

**Время выполнения**: 2-5 секунд

---

## Шаг 6: Понимание результата

Вы увидите:

```
[Шаг 1: Classification]
  Кандидаты: GPT-4 Classifier, Claude Classifier, Local Classifier
  Выбран: GPT-4 Classifier (вероятность: 0.654)
  
[Выполнение]
  [GPT-4 Classifier] Processed: Классифицировать этот текст...
  
[Результат]
  Категория: Positive
```

**Что означает**:
- **Шаг 1**: GraphArchitect нашел, что нужен Classifier
- **Кандидаты**: 3 инструмента могут выполнить
- **Выбран**: GPT-4 с вероятностью 65.4% (softmax)
- **Результат**: Обработанный текст

---

## Шаг 7: Проверка через API

Откройте новый терминал:

```bash
# Health check
curl http://localhost:8000/api/health
```

**Ожидаемый вывод**:
```json
{
  "status": "success",
  "data": {
    "version": "3.0.0",
    "status": "online",
    "grapharchitect_enabled": true,
    "features": {
      "real_algorithms": true,
      "softmax_selection": true,
      "training": true,
      "nli": true
    }
  }
}
```

Попробуйте самостоятельно:
```bash
# Список инструментов
curl http://localhost:8000/api/agents-library
```

---

## Troubleshooting

### Ошибка: "Database not found"

**Решение**:
```bash
python db_manager.py init
python db_manager.py load_agents
```

### Ошибка: "Port already in use"

**Решение**: Сервер автоматически найдет свободный порт (8000-8010)

### Ошибка: "GraphArchitect not available"

**Решение**:
```bash
set PYTHONPATH=..;%PYTHONPATH%
python main.py
```

---

## Итоги

### Вы успешно

- Установили GraphArchitect
- Инициализировали базу данных
- Запустили сервер
- Выполнили первую задачу
- Проверили API

### Ключевые моменты

- GraphArchitect использует граф инструментов
- Каждый запрос проходит через NLI → Поиск → Выбор → Выполнение
- Система обучается на каждом выполнении
- Все данные сохраняются в SQLite

### Следующие шаги

1. **Изучите основные концепции**: [02_basic_concepts.md](02_basic_concepts.md)
2. **Создайте свой workflow**: [03_first_workflow.md](03_first_workflow.md)
3. **Поймите инструменты**: [04_understanding_tools.md](04_understanding_tools.md)

---

**Время выполнения**: 5 минут  
**Следующий туториал**: [Основные концепции](02_basic_concepts.md)