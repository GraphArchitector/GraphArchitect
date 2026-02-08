"""
Service layer with business logic.
Handles chat, workflow, and document operations.
"""
import asyncio
import uuid
import logging
from typing import AsyncGenerator, List, Optional
from datetime import datetime
import aiofiles
import os

from models import (
    Agent, WorkflowChain, MessageRequest, MessageChunk,
    DocumentInfo, WorkflowCreateRequest, WorkflowCreateResponse,
    MessageResponse
)
from repository import get_repository
import config

# Configure logging
logger = logging.getLogger(__name__)

# GraphArchitect integration
try:
    from grapharchitect_bridge import get_bridge, is_bridge_available, AgentTool
    GRAPHARCHITECT_ENABLED = True
    logger.info("GraphArchitect integration activated")
except ImportError as e:
    GRAPHARCHITECT_ENABLED = False
    logger.warning(f"GraphArchitect not available: {e}")
    logger.warning("Using simulation mode")


class ChatService:
    """Service for chat and message operations."""
    
    def __init__(self):
        self.repo = get_repository()
    
    async def create_workflow(self, request: WorkflowCreateRequest) -> WorkflowChain:
        """
        Create agent workflow chain for chat.
        
        Args:
            request: Workflow creation request
            
        Returns:
            Created workflow chain
        """
        from workflow_templates import get_workflow_template
        
        # Select algorithm
        workflow = get_workflow_template(request.planning_algorithm) or get_workflow_template("yen_5")
        
        workflow.chat_id = request.chat_id
        workflow.files = request.files or []
        
        # Save to database
        self.repo.save_workflow(workflow)
        
        return workflow
    
    async def get_workflow(self, chat_id: str) -> Optional[WorkflowChain]:
        """Get workflow chain for chat."""
        return self.repo.get_workflow(chat_id)
    
    async def generate_graph_architecture_stream(self, request: WorkflowCreateRequest) -> AsyncGenerator[MessageChunk, None]:
        """
        Stream graph architecture generation phases.
        
        Args:
            request: Workflow creation request
            
        Yields:
            MessageChunk objects with generation progress
        """
        from workflow_templates import get_workflow_template
        
        workflow = get_workflow_template(request.planning_algorithm) or get_workflow_template("yen_5")
        
        # Workflow info
        yield MessageChunk(
            type="workflow_info",
            metadata={
                "name": workflow.name,
                "steps": [{"id": s.id, "name": s.name} for s in workflow.steps]
            }
        )

        top_k = 5
        if "3" in workflow.name: 
            top_k = 3
        elif "10" in workflow.name: 
            top_k = 10

        phases = [
            ("knn", "Searching architectures in k-NN..."),
            ("graph_algo", f"Generating {top_k} variants ({workflow.name})"),
            ("llm_refine", f"LLM synthesis from Top-{top_k} paths")
        ]

        for phase_id, phase_name in phases:
            yield MessageChunk(type="gen_phase_start", phase_id=phase_id, content=phase_name)
            for i in range(5):
                await asyncio.sleep(0.3)
                yield MessageChunk(type="gen_progress", phase_id=phase_id, progress=(i+1)*20)
            yield MessageChunk(type="gen_phase_complete", phase_id=phase_id)
            await asyncio.sleep(0.2)

    async def process_full_workflow_stream(
        self, 
        request: MessageRequest,
        use_rewoo: bool = None
    ) -> AsyncGenerator[MessageChunk, None]:
        """
        Full workflow cycle with streaming: Design -> Selection -> Execution.
        
        Args:
            request: Message request
            
        Yields:
            MessageChunk objects with workflow progress
        """
        logger.debug(f"Processing workflow with algorithm: {request.planning_algorithm}")
        
        # Определяем use_rewoo
        use_rewoo_flag = use_rewoo if use_rewoo is not None else getattr(request, 'use_rewoo', False)
        
        # Check: Use GraphArchitect or simulation
        if GRAPHARCHITECT_ENABLED and is_bridge_available():
            # REAL execution through GraphArchitect
            logger.info(f"Mode: GraphArchitect (real algorithms, ReWOO={use_rewoo_flag})")
            
            bridge = get_bridge()
            
            async for chunk in bridge.execute_task_streaming(
                message=request.message,
                input_data=request.message,
                algorithm=request.planning_algorithm,
                top_k=5,
                use_rewoo=use_rewoo_flag,
                user_priority=getattr(request, 'user_priority', 'balanced'),
                max_cost=getattr(request, 'max_cost', None),
                max_time=getattr(request, 'max_time', None)
            ):
                yield chunk
        
        else:
            # SIMULATION (fallback if GraphArchitect not available)
            logger.info("Mode: Simulation (GraphArchitect not available)")
            
            from workflow_templates import get_workflow_template
            import random
            
            repo = self.repo

            # 1. PREPARATION (Architecture generation)
            async for chunk in self.generate_graph_architecture_stream(
                WorkflowCreateRequest(
                    chat_id=request.chat_id, 
                    user_message=request.message, 
                    planning_algorithm=request.planning_algorithm,
                    request_type="text",
                    files=request.files
                )
            ):
                yield chunk

            # Get workflow for execution
            workflow = get_workflow_template(request.planning_algorithm) or get_workflow_template("yen_5")
            logger.debug(f"Selected template name: {workflow.name}")
            
            await asyncio.sleep(0.5)

            # 2. STEP EXECUTION (Selection + Execution)
            for step in workflow.steps:
                # STEP START
                yield MessageChunk(
                    type="step_started", 
                    step_id=step.id, 
                    metadata={"name": step.name, "candidates": step.candidate_agents}
                )
                await asyncio.sleep(0.3)

                # AGENT SELECTION (Competition)
                candidates = [repo.get_agent(aid) for aid in step.candidate_agents if repo.get_agent(aid)]
                scores = {c.id: 0 for c in candidates}
                
                # Accelerated selection
                for p in range(0, 101, 10):
                    await asyncio.sleep(0.12) 
                    for c in candidates:
                        # Simulate confidence growth
                        scores[c.id] = round(random.uniform(0.6, 0.95) if p < 80 else random.uniform(0.85, 0.99), 3)
                        yield MessageChunk(type="agent_progress", agent_id=c.id, progress=p, step_id=step.id)
                    
                    yield MessageChunk(
                        type="agent_score_updated", 
                        step_id=step.id,
                        metadata={"agents": [{"agentId": cid, "score": s} for cid, s in scores.items()]}
                    )

                winner = max(candidates, key=lambda c: scores[c.id])
                yield MessageChunk(type="agent_selected", agent_id=winner.id, step_id=step.id, score=scores[winner.id])
                
                await asyncio.sleep(0.8)

                # AGENT EXECUTION
                actions = ["Analyzing context...", "Generating solution...", "Validating result..."]
                for i, action in enumerate(actions):
                    await asyncio.sleep(0.5)
                    progress = int(((i+1)/len(actions))*100)
                    yield MessageChunk(type="agent_executing", agent_id=winner.id, step_id=step.id, progress=progress, content=action)

                yield MessageChunk(type="step_completed", step_id=step.id)
                await asyncio.sleep(0.4)

            # 3. FINAL TEXT
            final_text = f"Graph successfully executed using {workflow.name} algorithm."
            yield MessageChunk(type="text", content=final_text)
    
    async def process_message_stream(self, request: MessageRequest) -> AsyncGenerator[MessageChunk, None]:
        """
        Process message with streaming response.
        
        Args:
            request: Message request
            
        Yields:
            MessageChunk objects with response
        """
        async for chunk in self.process_full_workflow_stream(request):
            yield chunk
    
    async def process_message(self, request: MessageRequest) -> MessageResponse:
        """
        Process message without streaming.
        
        Args:
            request: Message request
            
        Returns:
            Message response
        """
        start_time = datetime.now()
        
        # Get workflow
        workflow = self.repo.get_workflow(request.chat_id)
        if not workflow:
            create_req = WorkflowCreateRequest(
                chat_id=request.chat_id,
                request_type="text",
                user_message=request.message,
                files=request.files
            )
            workflow_resp = await self.create_workflow(create_req)
            workflow = workflow_resp.workflow
        
        # Simulate processing
        await asyncio.sleep(1)
        
        # Form response
        response_text = f"Processed by {len(workflow.agents)} agents. Result ready."
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return MessageResponse(
            chat_id=request.chat_id,
            message=request.message,
            response_type="text",
            response_data=response_text,
            workflow_used=workflow.agents,
            processing_time=processing_time
        )


class DocumentService:
    """Service for document operations."""
    
    def __init__(self, upload_dir: str = None):
        self.repo = get_repository()
        self.upload_dir = upload_dir or str(config.UPLOAD_DIR)
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def save_document(
        self, 
        chat_id: str, 
        file: bytes,
        filename: str,
        content_type: str
    ) -> DocumentInfo:
        """
        Save document to storage and database.
        
        Args:
            chat_id: Chat identifier
            file: File bytes
            filename: Original filename
            content_type: MIME type
            
        Returns:
            Document information
        """
        # Generate unique ID
        document_id = str(uuid.uuid4())
        
        # Save file
        file_ext = os.path.splitext(filename)[1]
        saved_filename = f"{document_id}{file_ext}"
        file_path = os.path.join(self.upload_dir, saved_filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file)
        
        # Create database record
        document = DocumentInfo(
            document_id=document_id,
            chat_id=chat_id,
            filename=filename,
            content_type=content_type,
            size=len(file),
            path=file_path
        )
        
        self.repo.save_document(document)
        
        # Create chat if not exists
        if not self.repo.get_chat(chat_id):
            self.repo.create_chat(chat_id)
        
        return document
    
    async def get_documents(self, chat_id: str) -> List[DocumentInfo]:
        """Get all documents for chat."""
        return self.repo.get_documents(chat_id)
    
    async def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Get document information."""
        return self.repo.get_document(document_id)
