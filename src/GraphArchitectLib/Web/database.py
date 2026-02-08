"""
SQLite3 database for GraphArchitect Web API.

Tables:
- agents - tool library (instead of hardcode)
- workflows - execution chains
- chats - chat information
- documents - uploaded documents
- executions - execution history
- feedbacks - feedback for training
- tool_metrics - tool metrics
"""

import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Database:
    """Класс для работы с SQLite базой данных"""
    
    def __init__(self, db_path: str = "grapharchitect.db"):
        """
        Инициализация БД.
        
        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для работы с соединением"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Для доступа к колонкам по имени
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Создать таблицы если их нет"""
        logger.info(f"Initializing SQLite database: {self.db_path}")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица агентов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    icon TEXT,
                    color TEXT,
                    specialization TEXT,
                    capabilities TEXT,  -- JSON array
                    cost REAL DEFAULT 0.0,
                    metrics TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица workflows
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    chat_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    request_type TEXT,
                    steps TEXT,  -- JSON array
                    agents TEXT,  -- JSON array (для обратной совместимости)
                    files TEXT,  -- JSON array
                    current_step_index INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица чатов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица документов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    size INTEGER,
                    path TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
                )
            """)
            
            # Таблица истории выполнений (для обучения)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    chat_id TEXT,
                    task_description TEXT,
                    input_format TEXT,
                    output_format TEXT,
                    algorithm_used TEXT,
                    status TEXT,  -- COMPLETED, FAILED
                    selected_tools TEXT,  -- JSON array
                    gradient_traces TEXT,  -- JSON array
                    result TEXT,
                    total_time REAL,
                    total_cost REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE SET NULL
                )
            """)
            
            # Таблица обратной связи
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedbacks (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    execution_id TEXT,
                    source TEXT,  -- USER, AUTO_CRITIC, SYSTEM
                    quality_score REAL NOT NULL,
                    success INTEGER,  -- 0 or 1
                    comment TEXT,
                    detailed_scores TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
                )
            """)
            
            # Таблица метрик инструментов (агрегированные данные)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_metrics (
                    agent_id TEXT PRIMARY KEY,
                    tool_name TEXT,
                    reputation REAL DEFAULT 0.5,
                    mean_cost REAL DEFAULT 1.0,
                    mean_time REAL DEFAULT 1.0,
                    training_sample_size INTEGER DEFAULT 1,
                    variance_estimate REAL DEFAULT 1.0,
                    quality_scores TEXT,  -- JSON array
                    capabilities_embedding TEXT,  -- JSON array
                    last_training_date TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
                )
            """)
            
            # Индексы для ускорения запросов
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_chat_id 
                ON documents(chat_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_chat_id 
                ON executions(chat_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_created_at 
                ON executions(created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedbacks_task_id 
                ON feedbacks(task_id)
            """)
            
            conn.commit()
            
            logger.info("Database tables created/verified")
    
    def insert_default_agents(self):
        """
        Вставить дефолтных агентов в БД.
        
        Вызывается один раз при первом запуске.
        """
        from models import Agent
        
        logger.info("Loading default tools to database...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Проверяем сколько агентов уже в БД
            default_agents = self._get_default_agents()
            
            # Вставляем отсутствующих агентов (INSERT OR IGNORE)
            inserted = 0
            for agent in default_agents:
                cursor.execute("""
                    INSERT OR IGNORE INTO agents 
                    (id, name, type, icon, color, specialization, capabilities, cost, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent.id,
                    agent.name,
                    agent.type,
                    agent.icon,
                    agent.color,
                    agent.specialization,
                    json.dumps(agent.capabilities),
                    agent.cost,
                    json.dumps(agent.metrics)
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            
            conn.commit()
            
            cursor.execute("SELECT COUNT(*) FROM agents")
            total = cursor.fetchone()[0]
            
            if inserted > 0:
                logger.info(f"Added {inserted} new tools (total: {total})")
            else:
                logger.info(f"Database has {total} tools, all up to date")
    
    def _get_default_agents(self):
        """Получить список дефолтных агентов для первоначальной загрузки."""
        from models import Agent
        
        return [
            # Classifiers
            Agent(id="agent-classifier-gpt4", name="GPT-4 Classifier", type="classification", icon="C1", color="#10b981", 
                  specialization="High accuracy classification", capabilities=["advanced_nlp"], cost=0.03, 
                  metrics={"avgResponseTime": 2800, "avgScore": 0.98}),
            Agent(id="agent-classifier-claude", name="Claude Classifier", type="classification", icon="C2", color="#6366f1",
                  specialization="Deep context understanding", capabilities=["reasoning"], cost=0.02,
                  metrics={"avgResponseTime": 3200, "avgScore": 0.95}),
            Agent(id="agent-classifier-local", name="Local Classifier", type="classification", icon="C3", color="#8b5cf6",
                  specialization="Fast local processing", capabilities=["privacy"], cost=0.001,
                  metrics={"avgResponseTime": 1200, "avgScore": 0.78}),
            Agent(id="agent-classifier-fast", name="Fast Classifier", type="classification", icon="C4", color="#eab308",
                  specialization="Ultra-fast analysis", capabilities=["speed"], cost=0.005,
                  metrics={"avgResponseTime": 800, "avgScore": 0.72}),
            
            # Content generators  
            Agent(id="agent-responder-creative", name="Creative Responder", type="content_generation", icon="W1", color="#ec4899",
                  specialization="Creative responses", capabilities=["storytelling"], cost=0.025,
                  metrics={"avgResponseTime": 4200, "avgScore": 0.85}),
            Agent(id="agent-responder-formal", name="Formal Responder", type="content_generation", icon="W2", color="#3b82f6",
                  specialization="Professional tone", capabilities=["clarity"], cost=0.02,
                  metrics={"avgResponseTime": 3800, "avgScore": 0.88}),
            Agent(id="agent-responder-technical", name="Technical Responder", type="content_generation", icon="W3", color="#14b8a6",
                  specialization="Technical documentation", capabilities=["precision"], cost=0.022,
                  metrics={"avgResponseTime": 4100, "avgScore": 0.89}),
            Agent(id="agent-responder-friendly", name="Friendly Responder", type="content_generation", icon="W4", color="#f97316",
                  specialization="Friendly communication", capabilities=["empathy"], cost=0.018,
                  metrics={"avgResponseTime": 3500, "avgScore": 0.82}),
            
            # QA tools
            Agent(id="agent-qa-strict", name="Strict QA", type="quality_assurance", icon="Q1", color="#ef4444",
                  specialization="Strict quality control", capabilities=["validation"], cost=0.01,
                  metrics={"avgResponseTime": 2200, "avgScore": 0.99}),
            Agent(id="agent-qa-balanced", name="Balanced QA", type="quality_assurance", icon="Q2", color="#f59e0b",
                  specialization="Balanced review", capabilities=["fairness"], cost=0.008,
                  metrics={"avgResponseTime": 2000, "avgScore": 0.87}),
            Agent(id="agent-qa-fast", name="Fast QA", type="quality_assurance", icon="Q3", color="#10b981",
                  specialization="Quick validation", capabilities=["speed"], cost=0.005,
                  metrics={"avgResponseTime": 1000, "avgScore": 0.76}),
            
            # Parsers
            Agent(id="agent-parser-fast", name="Fast Parser", type="parsing", icon="P1", color="#6366f1",
                  specialization="Quick parsing", capabilities=["speed"], cost=0.002,
                  metrics={"avgResponseTime": 500, "avgScore": 0.80}),
            
            # Writers and editors
            Agent(id="agent-technical-writer", name="Technical Writer", type="writing", icon="W5", color="#8b5cf6",
                  specialization="Technical documentation and specs", capabilities=["technical_writing"], cost=0.025,
                  metrics={"avgResponseTime": 4500, "avgScore": 0.90}),
            Agent(id="agent-structured-outliner", name="Structured Outliner", type="planning", icon="P2", color="#f59e0b",
                  specialization="Creating structured outlines", capabilities=["planning", "organization"], cost=0.015,
                  metrics={"avgResponseTime": 2500, "avgScore": 0.89}),
            Agent(id="agent-style-checker", name="Style Checker", type="editing", icon="E1", color="#ec4899",
                  specialization="Style and grammar checking", capabilities=["style_analysis"], cost=0.012,
                  metrics={"avgResponseTime": 1800, "avgScore": 0.89}),
            Agent(id="agent-style-improver", name="Style Improver", type="editing", icon="E2", color="#14b8a6",
                  specialization="Style improvement suggestions", capabilities=["style_improvement"], cost=0.018,
                  metrics={"avgResponseTime": 3200, "avgScore": 0.87}),
            
            # Analysis and reporting
            Agent(id="agent-summary-reporter", name="Summary Reporter", type="reporting", icon="R1", color="#3b82f6",
                  specialization="Creating summary reports", capabilities=["summarization", "reporting"], cost=0.020,
                  metrics={"avgResponseTime": 3500, "avgScore": 0.84}),
            Agent(id="agent-trend-analyzer", name="Trend Analyzer", type="research", icon="A1", color="#10b981",
                  specialization="Analyzing trends and patterns", capabilities=["analysis", "insights"], cost=0.028,
                  metrics={"avgResponseTime": 5000, "avgScore": 0.86}),
            
            # Data processing
            Agent(id="agent-web-scraper", name="Web Scraper", type="code_analysis", icon="D1", color="#6366f1",
                  specialization="Web data extraction", capabilities=["scraping", "data_extraction"], cost=0.010,
                  metrics={"avgResponseTime": 2000, "avgScore": 0.83}),
            
            # Промежуточные инструменты (для многошаговых цепочек)
            Agent(id="agent-analyzer", name="Text Analyzer", type="analysis", icon="A1", color="#8b5cf6",
                  specialization="Analyzing text content", capabilities=["analysis"], cost=0.012,
                  metrics={"avgResponseTime": 2000, "avgScore": 0.88}),
            Agent(id="agent-categorizer", name="Categorizer", type="classification", icon="C5", color="#14b8a6",
                  specialization="Categorizing content", capabilities=["categorization"], cost=0.010,
                  metrics={"avgResponseTime": 1800, "avgScore": 0.86}),
            
            # Множество QA агентов (для конкуренции)
            Agent(id="agent-universal-processor", name="Universal Processor", type="universal", icon="U1", color="#9333ea",
                  specialization="Universal text processing", capabilities=["any"], cost=0.015,
                  metrics={"avgResponseTime": 2500, "avgScore": 0.80}),
            Agent(id="agent-general-qa", name="General QA", type="qa", icon="Q4", color="#06b6d4",
                  specialization="General question answering", capabilities=["qa"], cost=0.020,
                  metrics={"avgResponseTime": 3000, "avgScore": 0.85}),
            Agent(id="agent-fast-qa", name="Fast QA 2", type="qa", icon="Q5", color="#10b981",
                  specialization="Fast question answering", capabilities=["speed"], cost=0.008,
                  metrics={"avgResponseTime": 1500, "avgScore": 0.76}),
            Agent(id="agent-accurate-qa", name="Accurate QA", type="qa", icon="Q6", color="#ef4444",
                  specialization="High accuracy QA", capabilities=["accuracy"], cost=0.030,
                  metrics={"avgResponseTime": 4000, "avgScore": 0.92}),
            Agent(id="agent-balanced-processor", name="Balanced Processor", type="universal", icon="U2", color="#f59e0b",
                  specialization="Balanced processing", capabilities=["balance"], cost=0.018,
                  metrics={"avgResponseTime": 2800, "avgScore": 0.83}),
            Agent(id="agent-smart-qa", name="Smart QA", type="qa", icon="Q7", color="#ec4899",
                  specialization="Smart question answering", capabilities=["smart"], cost=0.025,
                  metrics={"avgResponseTime": 3500, "avgScore": 0.89}),
            
            # Генерация изображений (text|question -> image|answer)
            Agent(id="agent-image-gemini", name="Gemini Image Gen", type="image_generation", icon="IG", color="#4285f4",
                  specialization="Image generation via Gemini", capabilities=["image_generation", "diagrams", "schemas"],
                  cost=0.04,
                  metrics={"avgResponseTime": 8000, "avgScore": 0.85}),
            Agent(id="agent-image-gpt", name="GPT Image Gen", type="image_generation", icon="I2", color="#10a37f",
                  specialization="Image generation via GPT", capabilities=["image_generation", "illustrations"],
                  cost=0.05,
                  metrics={"avgResponseTime": 10000, "avgScore": 0.88}),
        ]
    
    def clear_all_data(self):
        """Очистить все таблицы (для тестирования)"""
        logger.warning("Clearing all data...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            tables = [
                "feedbacks",
                "executions", 
                "tool_metrics",
                "documents",
                "workflows",
                "chats",
                "agents"
            ]
            
            for table in tables:
                cursor.execute(f"DELETE FROM {table}")
            
            conn.commit()
            
            logger.info("All data cleared")


# Singleton instance
_database: Optional[Database] = None


def get_database(db_path: str = "grapharchitect.db", insert_default_agent = True) -> Database:
    """
    Получить экземпляр базы данных (singleton).
    
    Args:
        db_path: Путь к файлу БД
    
    Returns:
        Database instance
    """
    global _database
    
    if _database is None:
        _database = Database(db_path)
        
        if insert_default_agent:
            # При первом запуске загружаем агентов из agent_library
            _database.insert_default_agents()
    
    return _database
