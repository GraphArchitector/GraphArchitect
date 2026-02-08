"""
SQLite Repository for GraphArchitect Web API.

Replaces InMemoryRepository with persistent storage.
"""

import json
import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

from models import (
    WorkflowChain, WorkflowStep, Agent, DocumentInfo, ChatInfo,
    SelectionCriteria, CandidateProgress
)
from database import get_database


class SQLiteRepository:
    """
    Repository с хранением в SQLite.
    
    Реализует те же методы что и InMemoryRepository,
    но с персистентным хранилищем.
    """
    
    def __init__(self, db_path: str = "grapharchitect.db", insert_default_agent = True):
        """
        Инициализация репозитория.
        
        Args:
            db_path: Путь к файлу БД
        """
        self.db = get_database(db_path, insert_default_agent)
        logger.info(f"SQLiteRepository initialized ({db_path})")
    
    # ============== Работа с агентами ==============
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Получить агента по ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agents WHERE id = ?",
                (agent_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_agent(row)
    
    def get_all_agents(self) -> List[Agent]:
        """Получить всех агентов из БД"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agents ORDER BY name")
            rows = cursor.fetchall()
            
            return [self._row_to_agent(row) for row in rows]
    
    def save_agent(self, agent: Agent) -> Agent:
        """Сохранить или обновить агента"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO agents 
                (id, name, type, icon, color, specialization, capabilities, cost, metrics, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent.id,
                agent.name,
                agent.type,
                agent.icon,
                agent.color,
                agent.specialization,
                json.dumps(agent.capabilities),
                agent.cost,
                json.dumps(agent.metrics),
                datetime.now().isoformat()
            ))
            
            return agent
    
    def _row_to_agent(self, row) -> Agent:
        """Конвертировать строку БД в Agent"""
        return Agent(
            id=row['id'],
            name=row['name'],
            type=row['type'],
            icon=row['icon'] or "T",
            color=row['color'] or "#6366f1",
            specialization=row['specialization'],
            capabilities=json.loads(row['capabilities']) if row['capabilities'] else [],
            cost=row['cost'],
            metrics=json.loads(row['metrics']) if row['metrics'] else {}
        )
    
    # ============== Работа с Workflow ==============
    
    def save_workflow(self, workflow: WorkflowChain) -> WorkflowChain:
        """Сохранить цепочку агентов"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Сериализуем steps
            steps_json = json.dumps([
                {
                    'id': s.id,
                    'name': s.name,
                    'order': s.order,
                    'description': s.description,
                    'status': s.status,
                    'phase': s.phase,
                    'candidate_agents': s.candidate_agents,
                    'selected_agent_id': s.selected_agent_id,
                    'selection_criteria': {
                        'strategy': s.selection_criteria.strategy,
                        'timeout': s.selection_criteria.timeout
                    } if s.selection_criteria else None,
                    'result': s.result
                }
                for s in workflow.steps
            ])
            
            # Сериализуем agents
            agents_json = json.dumps([
                {
                    'id': a.id,
                    'name': a.name,
                    'icon': a.icon,
                    'color': a.color,
                    'type': a.type,
                    'specialization': a.specialization,
                    'capabilities': a.capabilities,
                    'cost': a.cost,
                    'metrics': a.metrics
                }
                for a in workflow.agents
            ])
            
            cursor.execute("""
                INSERT OR REPLACE INTO workflows 
                (chat_id, name, description, request_type, steps, agents, files, 
                 current_step_index, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow.chat_id,
                workflow.name,
                workflow.description,
                workflow.request_type,
                steps_json,
                agents_json,
                json.dumps(workflow.files),
                workflow.current_step_index,
                workflow.created_at.isoformat(),
                datetime.now().isoformat()
            ))
            
            # Обновляем активность чата
            self.update_chat_activity(workflow.chat_id)
            
            return workflow
    
    def get_workflow(self, chat_id: str) -> Optional[WorkflowChain]:
        """Получить цепочку агентов по ID чата"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM workflows WHERE chat_id = ?",
                (chat_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_workflow(row)
    
    def delete_workflow(self, chat_id: str) -> bool:
        """Удалить цепочку агентов"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM workflows WHERE chat_id = ?", (chat_id,))
            return cursor.rowcount > 0
    
    def _row_to_workflow(self, row) -> WorkflowChain:
        """Конвертировать строку БД в WorkflowChain"""
        # Десериализуем steps
        steps_data = json.loads(row['steps']) if row['steps'] else []
        steps = []
        
        for s in steps_data:
            selection_criteria = None
            if s.get('selection_criteria'):
                selection_criteria = SelectionCriteria(
                    strategy=s['selection_criteria'].get('strategy', 'best_quality_score'),
                    timeout=s['selection_criteria'].get('timeout', 10000)
                )
            
            step = WorkflowStep(
                id=s['id'],
                name=s['name'],
                order=s['order'],
                description=s.get('description'),
                status=s.get('status', 'pending'),
                phase=s.get('phase'),
                candidateAgents=s.get('candidate_agents', []),
                selectedAgentId=s.get('selected_agent_id'),
                selectionCriteria=selection_criteria,
                candidatesProgress=[],
                result=s.get('result')
            )
            steps.append(step)
        
        # Десериализуем agents
        agents_data = json.loads(row['agents']) if row['agents'] else []
        agents = [
            Agent(
                id=a['id'],
                name=a['name'],
                icon=a.get('icon', 'T'),
                color=a.get('color', '#6366f1'),
                type=a.get('type', 'general'),
                specialization=a.get('specialization'),
                capabilities=a.get('capabilities', []),
                cost=a.get('cost', 0.0),
                metrics=a.get('metrics', {})
            )
            for a in agents_data
        ]
        
        return WorkflowChain(
            chat_id=row['chat_id'],
            name=row['name'] or "Workflow",
            description=row['description'],
            steps=steps,
            currentStepIndex=row['current_step_index'],
            created_at=datetime.fromisoformat(row['created_at']),
            request_type=row['request_type'] or "text",
            files=json.loads(row['files']) if row['files'] else [],
            agents=agents
        )
    
    # ============== Работа с документами ==============
    
    def save_document(self, document: DocumentInfo) -> DocumentInfo:
        """Сохранить информацию о документе"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents 
                (document_id, chat_id, filename, content_type, size, path, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                document.document_id,
                document.chat_id,
                document.filename,
                document.content_type,
                document.size,
                document.path,
                document.uploaded_at.isoformat()
            ))
            
            # Обновляем активность чата
            self.update_chat_activity(document.chat_id)
            
            return document
    
    def get_documents(self, chat_id: str) -> List[DocumentInfo]:
        """Получить все документы чата"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE chat_id = ? ORDER BY uploaded_at DESC",
                (chat_id,)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_document(row) for row in rows]
    
    def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Получить документ по ID"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_document(row)
    
    def _row_to_document(self, row) -> DocumentInfo:
        """Конвертировать строку БД в DocumentInfo"""
        return DocumentInfo(
            document_id=row['document_id'],
            chat_id=row['chat_id'],
            filename=row['filename'],
            content_type=row['content_type'],
            size=row['size'],
            path=row['path'],
            uploaded_at=datetime.fromisoformat(row['uploaded_at'])
        )
    
    # ============== Работа с чатами ==============
    
    def create_chat(self, chat_id: str, title: Optional[str] = None) -> ChatInfo:
        """Создать новый чат"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT OR IGNORE INTO chats (chat_id, title, created_at, last_activity)
                VALUES (?, ?, ?, ?)
            """, (chat_id, title, now, now))
            
            # Получаем созданный чат
            return self.get_chat(chat_id)
    
    def get_chat(self, chat_id: str) -> Optional[ChatInfo]:
        """Получить информацию о чате"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chats WHERE chat_id = ?",
                (chat_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Получаем связанные данные
            workflow = self.get_workflow(chat_id)
            documents = self.get_documents(chat_id)
            
            return ChatInfo(
                chat_id=row['chat_id'],
                title=row['title'],
                created_at=datetime.fromisoformat(row['created_at']),
                last_activity=datetime.fromisoformat(row['last_activity']),
                workflow_chain=workflow,
                documents=documents
            )
    
    def update_chat_activity(self, chat_id: str):
        """Обновить время последней активности"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Создаем чат если не существует
            cursor.execute("SELECT 1 FROM chats WHERE chat_id = ?", (chat_id,))
            if not cursor.fetchone():
                self.create_chat(chat_id)
            else:
                cursor.execute("""
                    UPDATE chats SET last_activity = ? WHERE chat_id = ?
                """, (datetime.now().isoformat(), chat_id))
    
    def list_chats(self) -> List[ChatInfo]:
        """Получить список всех чатов"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chats ORDER BY last_activity DESC")
            rows = cursor.fetchall()
            
            chats = []
            for row in rows:
                chat = ChatInfo(
                    chat_id=row['chat_id'],
                    title=row['title'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_activity=datetime.fromisoformat(row['last_activity']),
                    workflow_chain=None,  # Не загружаем для списка (оптимизация)
                    documents=[]
                )
                chats.append(chat)
            
            return chats
    
    def delete_chat(self, chat_id: str) -> bool:
        """Удалить чат (CASCADE удалит связанные данные)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            return cursor.rowcount > 0
    
    # ============== Работа с историей выполнений ==============
    
    def save_execution(
        self,
        execution_id: str,
        task_id: str,
        chat_id: Optional[str],
        task_description: str,
        input_format: str,
        output_format: str,
        algorithm_used: str,
        status: str,
        selected_tools: List[str],
        gradient_traces: List[Dict],
        result: Any,
        total_time: float,
        total_cost: float
    ):
        """
        Сохранить историю выполнения для обучения.
        
        Это новая функциональность - не было в InMemoryRepository.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO executions 
                (execution_id, task_id, chat_id, task_description, input_format, 
                 output_format, algorithm_used, status, selected_tools, gradient_traces,
                 result, total_time, total_cost, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                task_id,
                chat_id,
                task_description,
                input_format,
                output_format,
                algorithm_used,
                status,
                json.dumps(selected_tools),
                json.dumps(gradient_traces, default=str),
                json.dumps(result, default=str),
                total_time,
                total_cost,
                datetime.now().isoformat()
            ))
    
    def get_executions(
        self,
        chat_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получить историю выполнений"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if chat_id:
                cursor.execute("""
                    SELECT * FROM executions 
                    WHERE chat_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (chat_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM executions 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # ============== Работа с обратной связью ==============
    
    def save_feedback(
        self,
        task_id: str,
        execution_id: Optional[str],
        source: str,
        quality_score: float,
        success: bool,
        comment: str = "",
        detailed_scores: Optional[Dict[str, float]] = None
    ):
        """Сохранить обратную связь"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedbacks 
                (task_id, execution_id, source, quality_score, success, comment, detailed_scores, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                execution_id,
                source,
                quality_score,
                1 if success else 0,
                comment,
                json.dumps(detailed_scores) if detailed_scores else None,
                datetime.now().isoformat()
            ))
    
    def get_feedbacks(self, task_id: str) -> List[Dict[str, Any]]:
        """Получить всю обратную связь для задачи"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM feedbacks WHERE task_id = ? ORDER BY created_at",
                (task_id,)
            )
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # ============== Работа с метриками инструментов ==============
    
    def save_tool_metrics(
        self,
        agent_id: str,
        tool_name: str,
        reputation: float,
        mean_cost: float,
        mean_time: float,
        training_sample_size: int,
        variance_estimate: float,
        quality_scores: List[float],
        capabilities_embedding: Optional[List[float]] = None
    ):
        """Сохранить метрики инструмента после обучения"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tool_metrics 
                (agent_id, tool_name, reputation, mean_cost, mean_time, 
                 training_sample_size, variance_estimate, quality_scores,
                 capabilities_embedding, last_training_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id,
                tool_name,
                reputation,
                mean_cost,
                mean_time,
                training_sample_size,
                variance_estimate,
                json.dumps(quality_scores),
                json.dumps(capabilities_embedding) if capabilities_embedding else None,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
    
    def get_tool_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Получить метрики инструмента"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM tool_metrics WHERE agent_id = ?",
                (agent_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return dict(row)
    
    def get_all_tool_metrics(self) -> List[Dict[str, Any]]:
        """Получить метрики всех инструментов"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tool_metrics 
                ORDER BY reputation DESC
            """)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # ============== Статистика ==============
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Получить статистику выполнений"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Общее количество
            cursor.execute("SELECT COUNT(*) FROM executions")
            total = cursor.fetchone()[0]
            
            # Успешные
            cursor.execute("SELECT COUNT(*) FROM executions WHERE status = 'COMPLETED'")
            completed = cursor.fetchone()[0]
            
            # Средние метрики
            cursor.execute("""
                SELECT 
                    AVG(total_time) as avg_time,
                    AVG(total_cost) as avg_cost
                FROM executions
                WHERE status = 'COMPLETED'
            """)
            row = cursor.fetchone()
            
            # Средняя оценка качества
            cursor.execute("""
                SELECT AVG(quality_score) as avg_quality
                FROM feedbacks
            """)
            quality_row = cursor.fetchone()
            
            return {
                "total_executions": total,
                "completed_executions": completed,
                "success_rate": completed / total if total > 0 else 0,
                "average_time": row['avg_time'] or 0,
                "average_cost": row['avg_cost'] or 0,
                "average_quality": quality_row['avg_quality'] or 0
            }


# Singleton instance
_repository: Optional[SQLiteRepository] = None


def get_sqlite_repository(db_path: str = "grapharchitect.db") -> SQLiteRepository:
    """
    Получить экземпляр SQLite репозитория (singleton).
    
    Args:
        db_path: Путь к файлу БД
    
    Returns:
        SQLiteRepository instance
    """
    global _repository
    
    if _repository is None:
        _repository = SQLiteRepository(db_path)
    
    return _repository
