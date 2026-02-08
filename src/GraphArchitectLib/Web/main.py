"""
Main FastAPI application for GraphArchitect Web API.
Provides REST API and WebSocket functionality.
"""
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import os
import socket
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import aiofiles

from dotenv import load_dotenv
load_dotenv()

from api_router import api_router
from models import MessageRequest
from services import ChatService
from repository import get_repository
from workflow_templates import get_all_templates
import config

# WebSocket manager
import socketio
from websocket_manager import sio

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GraphArchitect", 
    description="Multi-Agent System with Dynamic Workflow and Competitive Agent Selection",
    version=config.API_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API router
app.include_router(api_router)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# Initialize services
chat_service = ChatService()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_file_type(filename: str) -> str:
    """Get file type category from extension."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.pdf']:
        return 'pdf'
    elif ext in ['.doc', '.docx']:
        return 'document'
    elif ext in ['.txt']:
        return 'text'
    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        return 'image'
    elif ext in ['.mp3', '.wav', '.m4a']:
        return 'audio'
    elif ext in ['.zip', '.rar', '.7z']:
        return 'archive'
    else:
        return 'file'


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with chat interface."""
    templates_list = get_all_templates()
    repo = get_repository()
    agents = repo.get_all_agents()
    
    # Convert agents to simple format for frontend
    agents_list = [
        {
            "name": agent.name,
            "color": agent.color,
            "desc": "Available"
        }
        for agent in agents[:5]  # Show first 5
    ]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "agents": agents_list,
        "templates": templates_list
    })


@app.post("/api/set-api-key")
async def set_api_key(request: Request):
    """Установить OpenRouter API ключ из Web UI."""
    try:
        data = await request.json()
        key = data.get("key", "").strip()
        
        if key and len(key) > 10:
            os.environ["OPENROUTER_API_KEY"] = key
            logger.info("OpenRouter API key set from Web UI")
            
            # Переинициализируем NLI и ReWOO в bridge
            try:
                from grapharchitect_bridge import get_bridge, is_bridge_available, REWOO_AVAILABLE
                if is_bridge_available():
                    bridge = get_bridge()
                    
                    # NLI: переключаемся на LLM режим
                    bridge.nli = bridge._create_nli_service()
                    bridge._load_nli_examples()
                    logger.info(f"NLI reinitialized: {bridge.nli.__class__.__name__}")
                    
                    # ReWOO: создаём планировщик с API ключом
                    if REWOO_AVAILABLE and not bridge.rewoo_planner:
                        from grapharchitect.planning.rewoo_planner import ReWOOPlanner
                        bridge.rewoo_planner = ReWOOPlanner(gemini_api_key=key)
                        logger.info("ReWOO Planner initialized with API key")
            except Exception as e:
                logger.warning(f"Failed to reinitialize services: {e}")
            
            return {"status": "ok", "message": "API key set"}
        else:
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            return {"status": "cleared", "message": "API key cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/workflow-templates")
async def get_workflow_templates():
    """Get list of available workflow templates."""
    return {"templates": get_all_templates()}


@app.get("/api/agents-library")
async def get_agents_library():
    """Get library of all agents from database."""
    repo = get_repository()
    agents = repo.get_all_agents()
    return {
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "icon": agent.icon,
                "color": agent.color,
                "type": agent.type,
                "specialization": agent.specialization,
                "capabilities": agent.capabilities,
                "metrics": agent.metrics
            }
            for agent in agents
        ]
    }


@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload files to server."""
    uploaded = []

    # Ensure upload directory exists
    config.UPLOAD_DIR.mkdir(exist_ok=True)

    for file in files:
        # Save file
        file_path = config.UPLOAD_DIR / file.filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        uploaded.append({
            "name": file.filename,
            "size": len(content),
            "type": get_file_type(file.filename)
        })

    # Generate HTML for displaying files
    html = ""
    for file_info in uploaded:
        html += f"""
        <div class="file-item">
            <div class="file-icon">{file_info['type']}</div>
            <div class="file-details">
                <div class="file-name">{file_info['name']}</div>
                <div class="file-size">{format_file_size(file_info['size'])}</div>
            </div>
        </div>
        """

    if html:
        return HTMLResponse(f'<div class="uploaded-files-container">{html}</div>')
    return HTMLResponse("")


@app.post("/chat/stream")
async def chat_stream_gui(
        request: Request,
        message: str = Form(...),
        files: Optional[str] = Form(None)
):
    """Streaming chat response for GUI (HTML format)."""
    
    # Parse uploaded files
    file_list = json.loads(files) if files else []
    
    # Generate chat_id for GUI session
    chat_id = "gui_session_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    msg_request = MessageRequest(
        chat_id=chat_id,
        message=message,
        files=[f["name"] for f in file_list] if file_list else []
    )

    async def generate():
        # Initial message with files
        if file_list:
            yield '<div class="message assistant-message">'
            yield '<div class="message-content">'
            yield '<strong>Uploaded files:</strong><br><br>'
            for file_info in file_list:
                yield f'- {file_info["type"]} {file_info["name"]} ({format_file_size(file_info["size"])})<br>'
            yield '<br>---<br><br>'
        else:
            yield '<div class="message assistant-message"><div class="message-content">'
        
        # Get stream from service and convert to HTML
        async for chunk in chat_service.process_message_stream(msg_request):
            if chunk.type == "workflow":
                # Parse workflow and send only agents array
                workflow_data = json.loads(chunk.content)
                agents_json = json.dumps(workflow_data.get('agents', []))
                yield f'<span data-workflow-agents=\'{agents_json}\' style="display:none;"></span>'
            
            elif chunk.type == "agent_start":
                # Agent start marker
                yield f'<span data-agent-start="{chunk.agent_id}" style="display:none;"></span>'
                yield f'<strong>{chunk.content}</strong><br><br>'
            
            elif chunk.type == "agent_complete":
                # Agent complete marker
                yield f'<span data-agent-complete="{chunk.agent_id}" style="display:none;"></span>'
            
            elif chunk.type == "text":
                # Text content
                yield chunk.content
        
        yield '</div></div>'

    return StreamingResponse(generate(), media_type="text/html")


# Wrap FastAPI app in Socket.IO
combined_asgi_app = socketio.ASGIApp(sio, app, socketio_path='/socket.io')


def is_port_available(port: int) -> bool:
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def find_available_port(start: int = None, end: int = None) -> int:
    """Find available port in range."""
    start = start or config.PORT_START
    end = end or config.PORT_END
    
    for port in range(start, end + 1):
        if is_port_available(port):
            return port
    
    raise RuntimeError(f"No available port found in range {start}-{end}")


if __name__ == "__main__":
    import uvicorn
    
    # Ensure upload directory exists
    config.UPLOAD_DIR.mkdir(exist_ok=True)
    
    try:
        # Find available port
        port = find_available_port()
        
        logger.info(f"Starting server on port {port}")
        logger.info(f"Web interface: http://127.0.0.1:{port}")
        logger.info(f"API documentation: http://127.0.0.1:{port}/docs")
        logger.info(f"Socket.IO listening on /socket.io")
        
        uvicorn.run(
            combined_asgi_app, 
            host=config.HOST, 
            port=port, 
            log_level=config.LOG_LEVEL.lower()
        )
        
    except RuntimeError as e:
        logger.error(str(e))
        logger.error("Stop other processes or change port manually")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
