"""
Repository for data operations.
Single responsibility: data persistence.
"""

import logging
from typing import Optional, List
from models import WorkflowChain, Agent, DocumentInfo, ChatInfo

logger = logging.getLogger(__name__)


def get_repository():
    """
    Get repository instance.
    Always returns SQLiteRepository in production.
    
    Returns:
        Repository instance
    """
    try:
        from sqlite_repository import SQLiteRepository
        repo = SQLiteRepository()
        logger.info("Using SQLite repository")
        return repo
    except Exception as e:
        logger.error(f"Failed to initialize SQLite repository: {e}")
        raise RuntimeError("Database initialization failed. Run: python db_manager.py init")
