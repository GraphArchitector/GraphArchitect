"""
Скрипт создания БД стандартов коннекторов (по схеме из ТЗ).

Создает структуру базы данных для хранения стандартов коннекторов,
типов данных и семантических категорий.
"""

import sqlite3
import os
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "connectors_standards.db"

def create_database():
    """Создать базу данных и таблицы."""
    
    if DB_PATH.exists():
        logger.warning(f"Database {DB_PATH} already exists. Removing...")
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    logger.info("Creating tables...")
    
    # 1. Таблица subtypes (Подтипы данных)
    cursor.execute("""
    CREATE TABLE subtypes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subtype_name TEXT NOT NULL UNIQUE
    );
    """)
    
    # 2. Таблица semantic_categories (Семантические категории)
    # На схеме поле называется subtype_name, но логичнее category_name
    # Используем subtype_name для строгого соответствия схеме, но подразумеваем категорию
    cursor.execute("""
    CREATE TABLE semantic_categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subtype_name TEXT NOT NULL UNIQUE
    );
    """)
    
    # 3. Таблица knowledge_domains (Области знаний)
    # Исправлена опечатка knownlage -> knowledge
    cursor.execute("""
    CREATE TABLE knowledge_domains (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain_name TEXT NOT NULL UNIQUE
    );
    """)
    
    # 4. Таблица input_connectors (Входные коннекторы)
    cursor.execute("""
    CREATE TABLE input_connectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        
        complex_type TEXT NOT NULL CHECK (complex_type IN ('structured', 'file')),
        subtype_id INTEGER,
        properties TEXT NOT NULL,  -- JSON string
        
        semantic_category_id INTEGER,
        semantic_properties TEXT NOT NULL, -- JSON string
        
        knowledge_domain_id INTEGER,
        
        ti TEXT NOT NULL CHECK (ti IN ('текстовый', 'векторный')),
        tt TEXT NOT NULL CHECK (tt IN ('текстовый')),
        tc TEXT NOT NULL CHECK (tc IN ('текстовый', 'токены')),
        
        FOREIGN KEY (subtype_id) REFERENCES subtypes(id),
        FOREIGN KEY (semantic_category_id) REFERENCES semantic_categories(id),
        FOREIGN KEY (knowledge_domain_id) REFERENCES knowledge_domains(id)
    );
    """)
    
    # 5. Таблица output_connectors (Выходные коннекторы)
    cursor.execute("""
    CREATE TABLE output_connectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        
        complex_type TEXT NOT NULL CHECK (complex_type IN ('structured', 'file')),
        subtype_id INTEGER,
        properties TEXT NOT NULL, -- JSON string
        
        semantic_category_id INTEGER,
        semantic_properties TEXT NOT NULL, -- JSON string
        
        knowledge_domain_id INTEGER,
        
        FOREIGN KEY (subtype_id) REFERENCES subtypes(id),
        FOREIGN KEY (semantic_category_id) REFERENCES semantic_categories(id),
        FOREIGN KEY (knowledge_domain_id) REFERENCES knowledge_domains(id)
    );
    """)
    
    # 6. Таблица Tools (Инструменты)
    cursor.execute("""
    CREATE TABLE Tools (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        
        input_connector_id INTEGER NOT NULL,
        output_connector_id INTEGER NOT NULL,
        
        name TEXT NOT NULL,
        group_id TEXT NOT NULL,
        
        FOREIGN KEY (input_connector_id) REFERENCES input_connectors(id),
        FOREIGN KEY (output_connector_id) REFERENCES output_connectors(id)
    );
    """)
    
    conn.commit()
    logger.info("Tables created successfully.")
    
    # Наполнение справочников
    fill_dictionaries(cursor)
    
    conn.commit()
    conn.close()
    logger.info(f"Database created at {DB_PATH}")

def fill_dictionaries(cursor):
    """Наполнить справочные таблицы базовыми значениями."""
    logger.info("Filling dictionaries...")
    
    # Subtypes (из отчета)
    subtypes = [
        # Files
        'txt', 'json', 'pdf', 'jpg', 'png', 'wav', 'mp3', 'flac', 'ogg',
        # Structured
        'matrix', 'vector', 'tensor', 'text', 'image', 'sound', 'signal'
    ]
    cursor.executemany("INSERT INTO subtypes (subtype_name) VALUES (?)", [(s,) for s in subtypes])
    
    # Semantic Categories (из отчета)
    sem_categories = [
        'raw', 'specter', 'cepstrum', 'speech', 'question', 'answer', 
        'summary', 'reasoning', 'report', 'outline', 'category', 'validated',
        'findings', 'topic', 'article', 'draft', 'polished', 'code', 'analysis',
        'data', 'description', 'extracted'
    ]
    cursor.executemany("INSERT INTO semantic_categories (subtype_name) VALUES (?)", [(s,) for s in sem_categories])
    
    # Knowledge Domains
    domains = [
        'general', 'physics', 'math', 'biology', 'medicine', 'IT', 
        'finance', 'law', 'history', 'literature'
    ]
    cursor.executemany("INSERT INTO knowledge_domains (domain_name) VALUES (?)", [(d,) for d in domains])
    
    logger.info("Dictionaries filled.")

def add_example_tool(conn):
    """Добавить пример инструмента (GPT-4 Classifier)."""
    cursor = conn.cursor()
    
    # 1. Находим ID справочных значений
    cursor.execute("SELECT id FROM subtypes WHERE subtype_name = 'text'")
    subtype_id = cursor.fetchone()[0]
    
    cursor.execute("SELECT id FROM semantic_categories WHERE subtype_name = 'question'")
    sem_cat_in_id = cursor.fetchone()[0]
    
    cursor.execute("SELECT id FROM semantic_categories WHERE subtype_name = 'category'")
    sem_cat_out_id = cursor.fetchone()[0]
    
    cursor.execute("SELECT id FROM knowledge_domains WHERE domain_name = 'general'")
    domain_id = cursor.fetchone()[0]
    
    # 2. Создаем Input Connector (text|question)
    cursor.execute("""
        INSERT INTO input_connectors (
            complex_type, subtype_id, properties, 
            semantic_category_id, semantic_properties, 
            knowledge_domain_id, 
            ti, tt, tc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        'structured', subtype_id, '{"encoding": "utf-8"}',
        sem_cat_in_id, '{"language": "russian"}',
        domain_id,
        'текстовый', 'текстовый', 'текстовый'
    ))
    input_id = cursor.lastrowid
    
    # 3. Создаем Output Connector (text|category)
    cursor.execute("""
        INSERT INTO output_connectors (
            complex_type, subtype_id, properties, 
            semantic_category_id, semantic_properties, 
            knowledge_domain_id
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        'structured', subtype_id, '{"encoding": "utf-8"}',
        sem_cat_out_id, '{"format": "label"}',
        domain_id
    ))
    output_id = cursor.lastrowid
    
    # 4. Создаем Tool
    cursor.execute("""
        INSERT INTO Tools (
            input_connector_id, output_connector_id, 
            name, group_id
        ) VALUES (?, ?, ?, ?)
    """, (
        input_id, output_id, 
        "GPT-4 Classifier", "classifiers"
    ))
    
    conn.commit()
    logger.info("Example tool added.")

if __name__ == "__main__":
    create_database()
    
    # Добавляем пример
    conn = sqlite3.connect(DB_PATH)
    add_example_tool(conn)
    conn.close()
