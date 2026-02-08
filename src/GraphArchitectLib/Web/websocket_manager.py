"""
WebSocket Manager using Socket.IO for real-time communication.
"""
import socketio
import logging
from typing import Dict
from workflow_simulator import WorkflowSimulator
from workflow_templates import get_workflow_template
from models import WorkflowChain
from repository import get_repository
import asyncio

logger = logging.getLogger(__name__)


# Создаем Socket.IO сервер с логированием для отладки
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*', # Разрешаем все домены для разработки
    logger=True,
    engineio_logger=True # Включаем детальное логирование engine.io
)

# Активные симуляторы workflow
active_simulators: Dict[str, WorkflowSimulator] = {}


@sio.event
async def connect(sid, environ):
    """Клиент подключился"""
    logger.info(f"Client connected: {sid}")
    await sio.emit('connected', {'sid': sid}, room=sid)


@sio.event
async def disconnect(sid):
    """Клиент отключился"""
    logger.info(f"Client disconnected: {sid}")
    
    # Останавливаем симуляторы для этого клиента если есть
    to_remove = []
    for workflow_id, simulator in active_simulators.items():
        if workflow_id.startswith(sid):
            try:
                await simulator.stop()
            except Exception as e:
                logger.error(f"Error stopping simulator on disconnect: {e}")
            to_remove.append(workflow_id)
    
    for workflow_id in to_remove:
        del active_simulators[workflow_id]


@sio.event
async def start_workflow(sid, data):
    """Запустить выполнение workflow"""
    try:
        template_name = data.get('template', 'customer_support')
        chat_id = data.get('chat_id', f"{sid}_workflow")
        files = data.get('files', [])
        
        logger.info(f"Starting workflow: {template_name} for {sid} with {len(files)} files")
        
        # Получаем шаблон workflow
        workflow = get_workflow_template(template_name)
        if not workflow:
            await sio.emit('error', {
                'message': f'Template {template_name} not found'
            }, room=sid)
            return
        
        workflow.chat_id = chat_id
        workflow.files = files
        
        # Создаем callback для отправки событий
        async def emit_to_client(event_type: str, event_data: dict):
            await sio.emit(event_type, event_data, room=sid)
        
        # Создаем и запускаем симулятор
        simulator = WorkflowSimulator(workflow, emit_to_client)
        active_simulators[chat_id] = simulator
        
        # Отправляем информацию о workflow клиенту
        await sio.emit('workflow_info', {
            'workflowId': workflow.chat_id,
            'name': workflow.name,
            'description': workflow.description,
            'steps': [
                {
                    'id': step.id,
                    'name': step.name,
                    'order': step.order,
                    'description': step.description,
                    'candidateAgents': step.candidate_agents,
                    'strategy': step.selection_criteria.strategy
                }
                for step in workflow.steps
            ]
        }, room=sid)
        
        # Запускаем симуляцию в фоне
        asyncio.create_task(simulator.start())
        
    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        await sio.emit('error', {
            'message': f'Error starting workflow: {str(e)}'
        }, room=sid)


@sio.event
async def stop_workflow(sid, data):
    """Остановить выполнение workflow"""
    try:
        workflow_id = data.get('workflow_id')
        
        if workflow_id in active_simulators:
            simulator = active_simulators[workflow_id]
            await simulator.stop()  # Теперь async
            del active_simulators[workflow_id]
            
            logger.info(f"Workflow stopped: {workflow_id}")
        else:
            await sio.emit('error', {
                'message': f'Workflow {workflow_id} not found'
            }, room=sid)
    
    except Exception as e:
        logger.error(f"Error stopping workflow: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {
            'message': f'Error stopping workflow: {str(e)}'
        }, room=sid)


@sio.event
async def get_templates(sid, data=None):
    """Получить список доступных шаблонов"""
    from workflow_templates import get_all_templates
    
    templates = get_all_templates()
    await sio.emit('templates_list', {
        'templates': templates
    }, room=sid)


@sio.event
async def get_agents(sid, data=None):
    """Получить список всех агентов"""
    repo = get_repository()
    agents = repo.get_all_agents()
    
    agents_data = [
        {
            'id': agent.id,
            'name': agent.name,
            'icon': agent.icon,
            'color': agent.color,
            'type': agent.type,
            'specialization': agent.specialization,
            'capabilities': agent.capabilities,
            'metrics': agent.metrics
        }
        for agent in agents
    ]
    
    await sio.emit('agents_list', {
        'agents': agents_data
    }, room=sid)


# Socket.IO сервер готов для интеграции
# Экспортируем только sio, приложение будет создано в main.py
