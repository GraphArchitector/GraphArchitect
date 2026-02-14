"""
API Router - Чистое REST API
Может использоваться независимо от GUI
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import List, Optional
import json
import logging

logger = logging.getLogger(__name__)

from models import (
    MessageRequest, MessageResponse, MessageChunk, WorkflowCreateRequest,
    WorkflowCreateResponse, WorkflowChain, DocumentInfo,
    ApiResponse, ErrorResponse, ChatInfo
)
from services import ChatService, DocumentService


# Создаем роутер
api_router = APIRouter(prefix="/api", tags=["api"])

# Инициализируем сервисы
chat_service = ChatService()
document_service = DocumentService()


# ============== Workflow endpoints ==============

@api_router.post("/chat/{chat_id}/workflow")
async def create_workflow(
    chat_id: str,
    user_message: str = Form(...),
    request_type: str = Form("text"),
    files: Optional[str] = Form(None),
    planning_algorithm: str = Form("yen_5"),
    use_streaming: bool = Form(True)
):
    """
    Создать цепочку инструментов для чата (Генерация графа)
    """
    file_list = []
    if files:
        try:
            file_list = json.loads(files)
        except:
            file_list = [files] # Если не JSON, считаем как один ID

    req = WorkflowCreateRequest(
        chat_id=chat_id,
        request_type=request_type, # type: ignore
        user_message=user_message,
        files=file_list,
        planning_algorithm=planning_algorithm,
        use_streaming=use_streaming
    )
    
    if not use_streaming:
        return await chat_service.create_workflow(req)

    async def generate():
        async for chunk in chat_service.generate_graph_architecture_stream(req):
            yield chunk.model_dump_json() + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@api_router.get("/chat/{chat_id}/workflow", response_model=WorkflowChain)
async def get_workflow(chat_id: str):
    """
    Получить цепочку агентов для чата
    
    **Параметры:**
    - chat_id: ID чата
    
    **Возвращает:**
    - WorkflowChain - цепочка агентов
    
    **Ошибки:**
    - 404: Workflow не найден
    """
    workflow = await chat_service.get_workflow(chat_id)
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found for chat_id: {chat_id}"
        )
    
    return workflow


# ============== Message endpoints ==============

@api_router.post("/chat/{chat_id}/message/stream")
async def send_message_stream(
    chat_id: str,
    message: str = Form(...),
    files: Optional[str] = Form(None),
    planning_algorithm: str = Form("yen_5"),
    use_streaming: bool = Form(True),
    use_rewoo: bool = Form(False),
    user_priority: str = Form("balanced")
):
    """
    Унифицированный эндпоинт для работы с графом агентов.
    Выполняет: 
    1. Генерацию графа (3 фазы)
    2. Выполнение шагов (Выбор + Запуск)
    """
    try:
        file_list = json.loads(files) if files else []
        
        request = MessageRequest(
            chat_id=chat_id,
            message=message,
            files=file_list,
            planning_algorithm=planning_algorithm,
            use_streaming=use_streaming,
            use_rewoo=use_rewoo,
            user_priority=user_priority
        )
        
        if not use_streaming:
            # Логика для обычного (не стримингового) ответа
            response = await chat_service.process_message(request)
            return response

        async def generate():
            # Теперь сервис возвращает чанки для ВСЕХ этапов
            async for chunk in chat_service.process_full_workflow_stream(request):
                try:
                    json_str = chunk.model_dump_json()
                    
                    # Проверка размера чанка (для base64-изображений)
                    if len(json_str) > 10000000:  # 10MB
                        logger.warning(f"Chunk too large: {len(json_str)} bytes, truncating...")
                        # Отправляем сообщение об ошибке вместо огромного чанка
                        error_chunk = MessageChunk(
                            type="error",
                            content=f"Результат слишком большой для передачи ({len(json_str)} байт)"
                        )
                        yield error_chunk.model_dump_json() + "\n"
                    else:
                        yield json_str + "\n"
                
                except Exception as e:
                    logger.error(f"Error serializing chunk: {e}")
                    error_chunk = MessageChunk(type="error", content=f"Serialization error: {str(e)}")
                    yield error_chunk.model_dump_json() + "\n"
        
        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


@api_router.post("/chat/{chat_id}/message", response_model=MessageResponse)
async def send_message(
    chat_id: str,
    message: str = Form(...),
    files: Optional[List[str]] = Form(None)
):
    """
    Отправить сообщение без стриминга
    
    **Параметры:**
    - chat_id: ID чата
    - message: Текст сообщения
    - files: Список файлов (опционально)
    
    **Возвращает:**
    - MessageResponse с полным ответом
    
    **Примечание:**
    - Цепочка агентов подтягивается из БД по chat_id
    - Если цепочки нет, создается автоматически
    """
    try:
        request = MessageRequest(
            chat_id=chat_id,
            message=message,
            files=files or []
        )
        
        response = await chat_service.process_message(request)
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


# ============== Document endpoints ==============

@api_router.post("/chat/{chat_id}/document", response_model=DocumentInfo)
async def upload_document(
    chat_id: str,
    file: UploadFile = File(...)
):
    """
    Загрузить документ в чат
    
    **Параметры:**
    - chat_id: ID чата
    - file: Файл для загрузки
    
    **Возвращает:**
    - DocumentInfo с информацией о сохраненном документе
    
    **Примечание:**
    - Документ сохраняется в файловую систему
    - Метаданные сохраняются в БД
    """
    try:
        content = await file.read()
        
        document = await document_service.save_document(
            chat_id=chat_id,
            file=content,
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream"
        )
        
        return document
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@api_router.get("/chat/{chat_id}/documents", response_model=List[DocumentInfo])
async def get_documents(chat_id: str):
    """
    Получить все документы чата
    
    **Параметры:**
    - chat_id: ID чата
    
    **Возвращает:**
    - List[DocumentInfo] со списком документов
    """
    documents = await document_service.get_documents(chat_id)
    return documents


@api_router.get("/document/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """
    Получить информацию о документе
    
    **Параметры:**
    - document_id: ID документа
    
    **Возвращает:**
    - DocumentInfo с информацией о документе
    
    **Ошибки:**
    - 404: Документ не найден
    """
    document = await document_service.get_document(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    
    return document


# ============== Chat endpoints ==============

@api_router.get("/chat/{chat_id}", response_model=ChatInfo)
async def get_chat_info(chat_id: str):
    """
    Получить информацию о чате
    
    **Параметры:**
    - chat_id: ID чата
    
    **Возвращает:**
    - ChatInfo с полной информацией о чате
    
    **Ошибки:**
    - 404: Чат не найден
    """
    repo = chat_service.repo
    chat = repo.get_chat(chat_id)
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat not found: {chat_id}"
        )
    
    return chat


@api_router.get("/chats", response_model=List[ChatInfo])
async def list_chats():
    """
    Получить список всех чатов
    
    **Возвращает:**
    - List[ChatInfo] со списком всех чатов
    """
    repo = chat_service.repo
    chats = repo.list_chats()
    return chats


@api_router.delete("/chat/{chat_id}", response_model=ApiResponse)
async def delete_chat(chat_id: str):
    """
    Удалить чат и все связанные данные
    
    **Параметры:**
    - chat_id: ID чата
    
    **Возвращает:**
    - ApiResponse с результатом операции
    """
    repo = chat_service.repo
    success = repo.delete_chat(chat_id)
    
    if success:
        return ApiResponse(
            success=True,
            message=f"Chat {chat_id} deleted successfully"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat not found: {chat_id}"
        )


# ============== Training endpoints ==============

@api_router.post("/training/feedback")
async def submit_feedback(
    task_id: str = Form(...),
    quality_score: float = Form(...),
    comment: str = Form("")
):
    """
    Отправить обратную связь для обучения инструментов
    
    **Параметры:**
    - task_id: ID задачи (UUID)
    - quality_score: Оценка качества (0.0-1.0)
    - comment: Комментарий (опционально)
    
    **Возвращает:**
    - Результат обработки обратной связи
    """
    from training_service import get_training_service
    
    training_service = get_training_service()
    result = await training_service.submit_feedback(
        task_id=task_id,
        quality_score=quality_score,
        comment=comment
    )
    
    return result


@api_router.get("/training/statistics")
async def get_training_statistics():
    """
    Получить статистику обучения
    
    **Возвращает:**
    - Общая статистика обучения всех инструментов
    """
    from training_service import get_training_service
    
    training_service = get_training_service()
    stats = await training_service.get_statistics()
    
    return stats


@api_router.get("/training/tools/{agent_id}")
async def get_tool_metrics(agent_id: str):
    """
    Получить метрики конкретного инструмента
    
    **Параметры:**
    - agent_id: ID агента/инструмента
    
    **Возвращает:**
    - Детальные метрики инструмента
    """
    from training_service import get_training_service
    
    training_service = get_training_service()
    metrics = await training_service.get_tool_metrics(agent_id)
    
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool not found: {agent_id}"
        )
    
    return metrics


@api_router.get("/training/tools")
async def get_all_tools_metrics():
    """
    Получить метрики всех инструментов
    
    **Возвращает:**
    - Список метрик всех инструментов (отсортировано по репутации)
    """
    from training_service import get_training_service
    
    training_service = get_training_service()
    metrics = await training_service.get_all_tools_metrics()
    
    return metrics


@api_router.post("/training/train")
async def train_on_dataset(
    quality_threshold: float = Form(0.7)
):
    """
    Запустить обучение на накопленном датасете
    
    **Параметры:**
    - quality_threshold: Порог качества для фильтрации (0.0-1.0)
    
    **Возвращает:**
    - Результат обучения и статистику
    """
    from training_service import get_training_service
    
    training_service = get_training_service()
    result = await training_service.train_on_dataset(quality_threshold)
    
    return result


# ============== Health check ==============

@api_router.get("/health", response_model=ApiResponse)
async def health_check():
    """
    Проверка работоспособности API
    
    **Возвращает:**
    - ApiResponse со статусом API
    """
    try:
        from grapharchitect_bridge import is_bridge_available
        bridge_status = is_bridge_available()
    except:
        bridge_status = False
    
    return ApiResponse(
        success=True,
        message="API is healthy",
        data={
            "version": "3.0.0",
            "status": "online",
            "grapharchitect_enabled": bridge_status,
            "features": {
                "real_algorithms": bridge_status,
                "softmax_selection": bridge_status,
                "training": bridge_status,
                "nli": bridge_status
            }
        }
    )
