# webui.py - Multi-user version with session management

import logging
import asyncio
import os
import base64
import json
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.utils import utils
from src.utils.utils import MissingAPIKeyError
from src.agent.custom_agent import CustomAgent
from src.agent.custom_agent import logger as agent_logger 
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import BrowserContextConfig
from src.controller.custom_controller import CustomController
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextWindowSize
from src.session import SessionManager, UserSession
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global session manager
session_manager: SessionManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global session_manager
    
    # Startup
    max_sessions = int(os.getenv("MAX_CONCURRENT_SESSIONS", "10"))
    timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    
    session_manager = SessionManager(
        session_timeout_minutes=timeout_minutes,
        max_sessions=max_sessions
    )
    logger.info(f"Session manager started - Max sessions: {max_sessions}, Timeout: {timeout_minutes}min")
    
    yield
    
    # Shutdown
    if session_manager:
        await session_manager.shutdown()
    logger.info("Application shutdown complete")

app = FastAPI(lifespan=lifespan)

# Mount the 'static' directory to serve HTML, CSS, JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response Models
class CreateSessionRequest(BaseModel):
    """Request model for creating a session"""
    pass

class CreateSessionResponse(BaseModel):
    """Response model for session creation"""
    session_id: str
    status: str

class AgentRunRequest(BaseModel):
    """Request model for running an agent"""
    session_id: str
    task: str
    llm_provider: str = "google"  # Required: openai, google, anthropic, deepseek, etc.
    llm_model_name: str | None = None  # Optional: if not provided, uses default for provider
    llm_temperature: float = 0.6
    llm_base_url: str | None = None  # Optional: custom endpoint URL
    llm_api_key: str | None = None  # Optional: if not provided, uses environment variable

class AgentRunResponse(BaseModel):
    """Response model for agent run"""
    status: str
    session_id: str
    task: str | None = None

class SessionInfoResponse(BaseModel):
    """Response model for session information"""
    session_id: str
    created_at: str
    last_activity: str
    has_browser: bool
    has_context: bool
    has_agent: bool
    task_running: bool
    websocket_count: int

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    active_sessions: int
    max_sessions: int


class PlanRequest(BaseModel):
    task: str
    llm_provider: str = "groq"
    llm_model_name: str | None = None
    llm_api_key: str | None = None

class Step(BaseModel):
    id: int
    action: str
    description: str
    params: dict = {}

class ExecuteStepsRequest(BaseModel):
    session_id: str
    steps: List[Step]
    start_from: int = 0
    llm_provider: str = "groq"
    llm_model_name: str | None = None
    llm_api_key: str | None = None


async def send_socket_message_to_session(session: UserSession, message: dict):
    """Helper to send a JSON message to all WebSockets in a session."""
    dead_sockets = []
    for websocket in session.websockets:
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            dead_sockets.append(websocket)
    
    # Remove dead sockets
    for ws in dead_sockets:
        session.websockets.remove(ws)
        
      
        
async def run_agent_logic(config: AgentRunRequest, session: UserSession):
    """Run agent logic for a specific session"""
    original_info_handler = agent_logger.info
    
    try:
        # Define the function that will send logs to the websocket
        def log_to_socket(msg, *args, **kwargs):
            try:
                log_message = msg % args if args else str(msg)
                asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": log_message}))
            except Exception as e:
                asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": str(msg)}))

        # Monkey-patch the logger for this session
        agent_logger.info = log_to_socket

        # Validate API key
        env_var_name = f"{config.llm_provider.upper()}_API_KEY"
        llm_api_key = config.llm_api_key or os.getenv(env_var_name)
        
        if not llm_api_key:
            raise MissingAPIKeyError(provider=config.llm_provider, env_var=env_var_name)

        model_name = config.llm_model_name or utils.model_names.get(config.llm_provider, [""])[0]

        # Initialize LLM
        llm = utils.get_llm_model(
            provider=config.llm_provider, 
            model_name=model_name,
            temperature=config.llm_temperature, 
            base_url=config.llm_base_url, 
            api_key=llm_api_key
        )
        
        await send_socket_message_to_session(session, {"type": "log", "data": f"Creating isolated browser for session {session.session_id}..."})
        
        # **CRITICAL FIX**: Create completely isolated browser instance per session
        session.browser = CustomBrowser(config=BrowserConfig(
            headless=False
        ))
        
        # **CRITICAL FIX**: Create completely isolated browser context
        session.browser_context = await session.browser.new_context(
            config=BrowserContextConfig(
                no_viewport=False, 
                browser_window_size=BrowserContextWindowSize(width=1280, height=720),
                # **KEY FIX**: Unique downloads path per session
                save_downloads_path=f"/tmp/downloads_{session.session_id}"
            )
        )

        # Create agent for this session with isolated browser
        session.agent = CustomAgent(
            task=config.task, 
            llm=llm, 
            browser=session.browser,  # **KEY**: Use session-specific browser
            browser_context=session.browser_context,  # **KEY**: Use session-specific context
            controller=CustomController(), 
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt
        )
        
        agent_logger.info(f"Agent starting for session {session.session_id} with task: {config.task}")
        history = await session.agent.run(max_steps=int(os.getenv("MAX_STEPS_PER_TASK", "100")))
        
        final_result = history.final_result()
        await send_socket_message_to_session(session, {"type": "result", "data": final_result})
        logger.info(f"✅ Agent finished for session {session.session_id}. Final result: {final_result}")

    except Exception as e:
        logger.error(f"Error in agent execution for session {session.session_id}: {e}", exc_info=True)
        await send_socket_message_to_session(session, {"type": "error", "data": str(e)})
    finally:
        # Restore the original logger function
        agent_logger.info = original_info_handler
        
        await send_socket_message_to_session(session, {"type": "log", "data": f"Agent execution completed for session {session.session_id}."})
        logger.info(f"Agent execution completed for session {session.session_id}")

async def run_steps_logic(config: ExecuteStepsRequest, session: UserSession):
    """Execute steps sequentially with memory chain"""
    original_info_handler = agent_logger.info
    
    try:
        def log_to_socket(msg, *args, **kwargs):
            try:
                log_message = msg % args if args else str(msg)
                asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": log_message}))
            except Exception:
                asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": str(msg)}))

        agent_logger.info = log_to_socket

        # Initialize LLM
        llm = await _initialize_llm(
            config.llm_provider, config.llm_model_name, 0.6,
            None, config.llm_api_key
        )

        # Create browser if not exists
        if not session.browser:
            await send_socket_message_to_session(session, {"type": "log", "data": f"Creating browser for session {session.session_id}..."})
            
            session.browser = CustomBrowser(config=BrowserConfig(
                headless=False,
                extra_browser_args=[
                    f"--user-data-dir=/tmp/chrome_data_{session.session_id}",
                    f"--remote-debugging-port={9222 + hash(session.session_id) % 1000}"
                ]
            ))
            
            session.browser_context = await session.browser.new_context(
                config=BrowserContextConfig(
                    no_viewport=False, 
                    browser_window_size=BrowserContextWindowSize(width=1280, height=720)
                )
            )

        # Execute steps starting from specified index
        for i, step in enumerate(config.steps[config.start_from:], config.start_from):
            if session.stopped:
                break
                
            await send_socket_message_to_session(session, {
                "type": "step_start", 
                "data": {"step": i+1, "total": len(config.steps), "description": step.description}
            })
            
            # Create step-specific task with memory
            step_task = f"""Execute this specific step with context from previous steps:

Step {step.id}: {step.description}
Action: {step.action}
Parameters: {step.params}

Previous steps context (memory):
{session.step_memory}

Execute only this specific step. Be precise and focused."""

            # Create agent for this step
            step_agent = CustomAgent(
                task=step_task,
                llm=llm,
                browser=session.browser,
                browser_context=session.browser_context,
                controller=CustomController(),
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt
            )
            
            # Execute step
            step_result = await step_agent.run(max_steps=5)
            step_output = step_result.final_result() or f"Completed step {step.id}"
            
            # Update memory chain
            session.step_memory += f"Step {step.id}: {step.description} - Result: {step_output}\n"
            session.executed_steps.append({
                "step_id": step.id,
                "description": step.description,
                "result": step_output,
                "completed_at": datetime.now().isoformat()
            })
            session.current_step_index = i + 1
            
            await send_socket_message_to_session(session, {
                "type": "step_complete",
                "data": {"step": i+1, "result": step_output}
            })
            
            agent_logger.info(f"Completed step {step.id}: {step.description}")

        # Final result
        await send_socket_message_to_session(session, {
            "type": "result", 
            "data": f"Completed {len(config.steps)} steps. Session remains active for continuation."
        })
        
    except Exception as e:
        logger.error(f"Error in steps execution for session {session.session_id}: {e}", exc_info=True)
        await send_socket_message_to_session(session, {"type": "error", "data": str(e)})
    finally:
        agent_logger.info = original_info_handler
        # DO NOT clean up session - keep it persistent
        await send_socket_message_to_session(session, {
            "type": "log", 
            "data": "Steps execution completed - session remains active for more steps"
        })
        logger.info(f"Steps execution completed for session {session.session_id} - session remains active")
        
# async def run_agent_logic(config: AgentRunRequest, session: UserSession):
#     """Run agent logic for a specific session"""
#     original_info_handler = agent_logger.info
    
#     try:
#         # Define the function that will send logs to the websocket
#         def log_to_socket(msg, *args, **kwargs):
#             try:
#                 log_message = msg % args if args else str(msg)
#                 asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": log_message}))
#             except Exception as e:
#                 # If formatting fails, send the raw message
#                 asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": str(msg)}))

#         # Monkey-patch the logger for this session
#         agent_logger.info = log_to_socket

#         # Validate API key
#         env_var_name = f"{config.llm_provider.upper()}_API_KEY"
#         llm_api_key = config.llm_api_key or os.getenv(env_var_name)
        
#         if not llm_api_key:
#             raise MissingAPIKeyError(provider=config.llm_provider, env_var=env_var_name)

#         model_name = config.llm_model_name or utils.model_names.get(config.llm_provider, [""])[0]

#         # Initialize LLM
#         llm = utils.get_llm_model(
#             provider=config.llm_provider, 
#             model_name=model_name,
#             temperature=config.llm_temperature, 
#             base_url=config.llm_base_url, 
#             api_key=llm_api_key
#         )
        
#         await send_socket_message_to_session(session, {"type": "log", "data": "Browser starting..."})
        
#         # **CRITICAL FIX**: Create isolated browser for this session
#         session.browser = CustomBrowser(config=BrowserConfig(
#             headless=False,
#             extra_browser_args=[
#                 f"--user-data-dir=/tmp/chrome_session_{session.session_id}",  # Unique data dir
#                 "--no-first-run",
#                 "--disable-dev-shm-usage",
#                 "--no-sandbox"
#             ]
#         ))
        
#         # **CRITICAL FIX**: Create isolated context for this session
#         session.browser_context = await session.browser.new_context(
#             config=BrowserContextConfig(
#                 no_viewport=False, 
#                 browser_window_size=BrowserContextWindowSize(width=1280, height=720),
#                 save_downloads_path=f"/tmp/downloads_session_{session.session_id}"  # Unique downloads
#             )
#         )

#         # Create agent for this session
#         session.agent = CustomAgent(
#             task=config.task, 
#             llm=llm, 
#             browser=session.browser, 
#             browser_context=session.browser_context,
#             controller=CustomController(), 
#             system_prompt_class=CustomSystemPrompt,
#             agent_prompt_class=CustomAgentMessagePrompt
#         )
        
#         agent_logger.info(f"Agent starting for task: {config.task}")
#         history = await session.agent.run(max_steps=int(os.getenv("MAX_STEPS_PER_TASK", "100")))
        
#         final_result = history.final_result()
#         await send_socket_message_to_session(session, {"type": "result", "data": final_result})
#         logger.info(f"✅ Agent finished for session {session.session_id}. Final result: {final_result}")

#     except Exception as e:
#         logger.error(f"Error in agent execution for session {session.session_id}: {e}", exc_info=True)
#         await send_socket_message_to_session(session, {"type": "error", "data": str(e)})
#     finally:
#         # Restore the original logger function
#         agent_logger.info = original_info_handler
        
#         await send_socket_message_to_session(session, {"type": "log", "data": "Agent execution completed."})
#         logger.info(f"Agent execution completed for session {session.session_id}")


async def stream_browser_view(session: UserSession):
    """Periodically captures and sends screenshots for a specific session."""
    while (session.current_task and not session.current_task.done() and 
           session.browser_context and session.browser):
        try:
            if hasattr(session.browser_context, 'browser') and session.browser_context.browser:
                playwright_browser = session.browser_context.browser.playwright_browser
                if playwright_browser and playwright_browser.contexts:
                    pw_context = playwright_browser.contexts[0]
                    if pw_context and pw_context.pages:
                        page = next((p for p in reversed(pw_context.pages) if p.url != "about:blank"), None)
                        if page and not page.is_closed():
                            screenshot_bytes = await page.screenshot(type="jpeg", quality=70)
                            b64_img = base64.b64encode(screenshot_bytes).decode('utf-8')
                            await send_socket_message_to_session(session, {"type": "stream", "data": b64_img})
        except Exception as e:
            logger.debug(f"Screenshot capture failed for session {session.session_id}: {e}")
        
        await asyncio.sleep(0.5)

async def _initialize_llm(provider: str, model_name: str, temperature: float, base_url: str, api_key: str):
    """Initialize LLM - helper function"""
    if not provider or not model_name:
        return None
    try:
        return utils.get_llm_model(
            provider=provider,
            model_name=model_name or utils.model_names.get(provider, [""])[0],
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_index():
    """Serve the main HTML page"""
    return FileResponse('static/index.html')

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        active_sessions=session_manager.get_session_count(),
        max_sessions=session_manager.max_sessions
    )

@app.get("/api/providers", response_class=JSONResponse)
async def get_providers():
    """Get available LLM providers"""
    return utils.PROVIDER_DISPLAY_NAMES

@app.post("/api/session/create", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest = CreateSessionRequest()):
    """Create a new user session"""
    try:
        session_id = session_manager.create_session()
        return CreateSessionResponse(session_id=session_id, status="Session created successfully")
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=503, detail=f"Could not create session: {str(e)}")

@app.get("/api/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    session_info = session_manager.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionInfoResponse(**session_info)

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    success = await session_manager.delete_session(session_id)
    if success:
        return {"status": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/agent/run", response_model=AgentRunResponse)
async def start_agent_run(request: AgentRunRequest):
    """Start an agent run in a specific session"""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.current_task and not session.current_task.done():
        raise HTTPException(status_code=409, detail="Agent is already running in this session")
    
    # Start agent task for this session
    session.current_task = asyncio.create_task(run_agent_logic(request, session))
    
    return AgentRunResponse(
        status="Agent started successfully", 
        session_id=session.session_id,
        task=request.task
    )

@app.post("/api/agent/stop/{session_id}")
async def stop_agent(session_id: str):
    """Stop the currently running agent in a session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.current_task and not session.current_task.done():
        session.current_task.cancel()
        try:
            await session.current_task
        except asyncio.CancelledError:
            pass
        return {"status": "Agent stopped successfully"}
    else:
        return {"status": "No agent running in this session"}

@app.post("/api/agent/plan")
async def create_execution_plan(request: PlanRequest):
    """Generate step-by-step execution plan from task"""
    try:
        # Initialize LLM
        llm = await _initialize_llm(
            request.llm_provider, request.llm_model_name, 0.3,
            None, request.llm_api_key
        )
        if not llm:
            raise HTTPException(status_code=400, detail="Failed to initialize LLM")
        
        # Create planning prompt
        planning_prompt = f"""Break down this task into specific, actionable steps for a browser automation agent:

Task: {request.task}

Return ONLY a JSON array of steps in this exact format:
[
  {{"id": 1, "action": "go_to_url", "description": "Navigate to the website", "params": {{"url": "https://example.com"}}}},
  {{"id": 2, "action": "search", "description": "Search for the product", "params": {{"query": "product name"}}}},
  {{"id": 3, "action": "click_element", "description": "Click on specific element", "params": {{"selector": ".class-name"}}}}
]

Available actions: go_to_url, search, click_element, input_text, extract_content, scroll_down, wait, done
Each step should be specific and executable by a browser agent."""

        response = await llm.ainvoke([{"role": "user", "content": planning_prompt}])
        
        # Parse response
        import json
        from json_repair import repair_json
        
        content = response.content.replace("```json", "").replace("```", "").strip()
        content = repair_json(content)
        steps_data = json.loads(content)
        
        # Convert to Step objects
        steps = [Step(**step_data) for step_data in steps_data]
        
        return {"steps": [step.dict() for step in steps], "status": "success"}
        
    except Exception as e:
        logger.error(f"Error creating execution plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/execute-steps")
async def execute_steps(request: ExecuteStepsRequest):
    """Execute steps sequentially with memory chain"""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.current_task and not session.current_task.done():
        raise HTTPException(status_code=409, detail="Agent already running in this session")
    
    # Mark session as persistent
    session.persistent = True
    
    # Start step execution
    session.current_task = asyncio.create_task(run_steps_logic(request, session))
    
    return {
        "status": "Steps execution started",
        "session_id": session.session_id,
        "total_steps": len(request.steps),
        "starting_from": request.start_from
    }

@app.post("/api/agent/continue-steps")
async def continue_with_more_steps(request: ExecuteStepsRequest):
    """Add and execute more steps to existing session"""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.current_task and not session.current_task.done():
        raise HTTPException(status_code=409, detail="Agent already running in this session")
    
    # Continue from where we left off
    session.current_task = asyncio.create_task(run_steps_logic(request, session))
    
    return {
        "status": "Continuing with additional steps",
        "session_id": request.session_id,
        "previous_steps_completed": len(session.executed_steps),
        "new_steps": len(request.steps)
    }
    
@app.post("/api/session/{session_id}/cleanup")
async def cleanup_persistent_session(session_id: str):
    """Manually cleanup a persistent session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Force cleanup even if persistent
    session.persistent = False
    success = await session_manager.delete_session(session_id)
    
    if success:
        return {"status": "Session cleaned up successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to cleanup session")


@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get detailed session status"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "persistent": session.persistent,
        "has_browser": session.browser is not None,
        "has_context": session.browser_context is not None,
        "task_running": session.current_task is not None and not session.current_task.done(),
        "websocket_count": len(session.websockets),
        "executed_steps": len(session.executed_steps),
        "current_step_index": session.current_step_index,
        "last_activity": session.last_activity.isoformat(),
        "step_memory": session.step_memory[:200] + "..." if len(session.step_memory) > 200 else session.step_memory
    }
    
@app.websocket("/ws/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming data to a specific session"""
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    await websocket.accept()
    session.websockets.append(websocket)
    session.update_activity()  # Update activity when WebSocket connects
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connected", 
            "data": f"Connected to session {session_id}"
        }))
        
        # Keep connection alive and listen for disconnect
        while True:
            try:
                # Wait for any message (ping/pong or close)
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # No message received, just continue
                continue
            except Exception:
                # Client disconnected
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Remove WebSocket from session but DON'T cleanup session
        if websocket in session.websockets:
            session.websockets.remove(websocket)
        # DO NOT call session cleanup here for persistent sessions

# --- Main Execution ---
def main():
    """Main entry point"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7788"))
    
    logger.info(f"Starting multi-user Browser Use API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()



