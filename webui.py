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
    generate_analysis: bool = False
    export_formats: List[str] = []  
    enable_popup_killer: bool = False
    enable_captcha_avoidance: bool = False  
    force_duckduckgo: bool = True


async def _initialize_llm_helper(provider: str, model_name: str, api_key: str):
    """Helper to initialize LLM - separate from existing logic"""
    logger.info(f"🔧 Initializing LLM: provider={provider}, model={model_name}")
    
    env_var = f"{provider.upper()}_API_KEY"
    llm_api_key = api_key or os.getenv(env_var)
    
    if not llm_api_key:
        logger.error(f" Missing API key for {provider}")
        raise MissingAPIKeyError(provider=provider, env_var=env_var)
    
    model = model_name or utils.model_names.get(provider, [""])[0]
    logger.info(f" Using model: {model}")
    
    try:
        llm = utils.get_llm_model(
            provider=provider,
            model_name=model,
            temperature=0.6,
            api_key=llm_api_key
        )
        logger.info(f"✅ LLM initialization successful")
        return llm
    except Exception as e:
        logger.error(f"❌ LLM initialization failed: {e}")
        raise
        
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
        
# async def run_steps_logic(config: ExecuteStepsRequest, session: UserSession):
#     """Execute steps with intelligence, adaptation, and goal tracking"""
#     logger.info(f"🚀 Starting goal-oriented execution for session {session.session_id}")
    
#     # Import intelligence modules
#     from src.intelligence.step_analyzer import StepAnalyzer, AdaptivePlanner
#     from src.intelligence.goal_tracker import GoalTracker
    
#     # Initialize session state safely
#     if not hasattr(session, 'execution_context') or session.execution_context is None:
#         session.execution_context = {}
    
#     session.execution_context.setdefault('current_url', '')
#     session.execution_context.setdefault('variables', {})
#     session.execution_context.setdefault('step_results', [])
#     session.execution_context.setdefault('memory_chain', '')
#     session.execution_context.setdefault('adaptations_made', [])
    
#     if not hasattr(session, 'executed_steps'):
#         session.executed_steps = []
#     if not hasattr(session, 'current_step_index'):
#         session.current_step_index = 0
    
#     def log_to_socket(msg, *args, **kwargs):
#         log_message = msg % args if args else str(msg)
#         asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": log_message}))
    
#     try:
#         # Initialize LLM
#         llm = await _initialize_llm_helper(config.llm_provider, config.llm_model_name, config.llm_api_key)
#         log_to_socket("🧠 Intelligent agent with goal tracking initialized")
        
#         # Create browser if needed
#         if not session.browser:
#             log_to_socket("🌐 Creating browser...")
#             session.browser = CustomBrowser(config=BrowserConfig(headless=False))
#             session.browser_context = await session.browser.new_context(
#                 config=BrowserContextConfig(
#                     no_viewport=False, 
#                     browser_window_size=BrowserContextWindowSize(width=1280, height=720),
#                     save_downloads_path=f"/tmp/downloads_{session.session_id}"
#                 )
#             )
        
#         # Initialize Goal Tracker
#         if not hasattr(session, 'goal_tracker') or session.goal_tracker is None:
#             # Extract main goal from first step or overall context
#             main_goal = f"Complete task: {config.steps[0].description if config.steps else 'Execute steps'}"
#             session.goal_tracker = GoalTracker(main_goal)
            
#             # Decompose goal into sub-goals
#             await session.goal_tracker.decompose_goal(llm)
#             log_to_socket(f"🎯 Goal set: {main_goal}")
            
#             # Send goal information to frontend
#             await send_socket_message_to_session(session, {
#                 "type": "goal_initialized",
#                 "data": {
#                     "main_goal": main_goal,
#                     "sub_goals": session.goal_tracker.sub_goals,
#                     "progress": 0
#                 }
#             })
        
#         # Convert steps to mutable list for adaptation
#         current_steps = list(config.steps[config.start_from:])
#         i = 0
        
#         while i < len(current_steps):
#             step = current_steps[i]
#             step_dict = {"id": step.id, "action": step.action, "description": step.description, "params": step.params}
            
#             log_to_socket(f"🎯 Step {i+1}/{len(current_steps)}: {step.description}")
            
#             # GOAL TRACKING: Check alignment before step
#             await send_socket_message_to_session(session, {
#                 "type": "thinking", 
#                 "data": {"step_id": step.id, "phase": "goal_checking", "message": "Checking goal alignment..."}
#             })
            
#             alignment = await session.goal_tracker.check_alignment(
#                 session.execution_context, 
#                 step.description, 
#                 llm
#             )
            
#             # Send goal status update
#             await send_socket_message_to_session(session, {
#                 "type": "goal_status",
#                 "data": {
#                     "alignment": alignment.get('alignment'),
#                     "progress": alignment.get('progress_percentage', 0),
#                     "reasoning": alignment.get('reasoning', ''),
#                     "next_focus": alignment.get('next_focus', '')
#                 }
#             })
            
#             # Handle major deviations
#             if alignment.get('alignment') == 'major_deviation':
#                 log_to_socket(f"⚠️ Major deviation detected: {alignment.get('reasoning')}")
                
#                 correction = await session.goal_tracker.suggest_course_correction(
#                     alignment.get('reasoning', 'Unknown deviation'),
#                     session.execution_context,
#                     llm
#                 )
                
#                 await send_socket_message_to_session(session, {
#                     "type": "course_correction",
#                     "data": {
#                         "deviation_reason": alignment.get('reasoning'),
#                         "correction": correction,
#                         "urgency": correction.get('urgency', 'medium')
#                     }
#                 })
                
#                 log_to_socket(f"🔄 Course correction: {correction.get('action')}")
                
#                 # Apply correction if needed
#                 if correction.get('correction_type') == 'skip_step':
#                     log_to_socket("⏭️ Skipping step due to course correction")
#                     i += 1
#                     continue
            
#             # PHASE 1: PRE-STEP ANALYSIS (with goal context)
#             await send_socket_message_to_session(session, {
#                 "type": "thinking", 
#                 "data": {"step_id": step.id, "phase": "analyzing", "message": "Analyzing step with goal context..."}
#             })
            
#             step_analysis = await StepAnalyzer.analyze_before_step(step_dict, session.execution_context, llm, session.goal_tracker)
#             log_to_socket(f"🧠 Goal: {step_analysis.get('goal', 'Execute step')}")
            
#             # Send step start with analysis
#             await send_socket_message_to_session(session, {
#                 "type": "step_start", 
#                 "data": {
#                     "step_id": step.id,
#                     "step_number": i + 1,
#                     "total": len(current_steps),
#                     "description": step.description,
#                     "analysis": step_analysis,
#                     "goal_alignment": step_analysis.get('goal_alignment')
#                 }
#             })
            
#             # EXECUTE STEP
#             try:
#                 # Build intelligent context prompt with goal awareness
#                 context_prompt = f"""You are an intelligent browser agent working toward a specific goal.

# MAIN GOAL: {session.goal_tracker.main_goal}
# GOAL PROGRESS: {session.goal_tracker.current_progress}%

# CURRENT STEP:
# GOAL: {step_analysis.get('goal', step.description)}
# ACTION: {step.action}
# PARAMETERS: {json.dumps(step.params)}
# GOAL CONTRIBUTION: {step_analysis.get('goal_contribution', 'Unknown')}

# CONTEXT:
# {session.execution_context.get('memory_chain', '')[-400:]}

# EXPECTATIONS: {step_analysis.get('expectations', 'Unknown')}
# SUCCESS CRITERIA: {step_analysis.get('success_criteria', 'Step completes')}

# IMPORTANT: Stay focused on the main goal. If you encounter distractions (popups, ads, irrelevant content), ignore them and stay on track toward: {session.goal_tracker.main_goal}

# Execute this step intelligently while keeping the main goal in mind."""
                
#                 await send_socket_message_to_session(session, {
#                     "type": "thinking", 
#                     "data": {"step_id": step.id, "phase": "executing", "message": f"Executing: {step_analysis.get('goal', step.description)}"}
#                 })
                
#                 # Create and run step agent
#                 step_agent = CustomAgent(
#                     task=context_prompt,
#                     llm=llm,
#                     browser=session.browser,
#                     browser_context=session.browser_context,
#                     controller=CustomController(),
#                     system_prompt_class=CustomSystemPrompt,
#                     agent_prompt_class=CustomAgentMessagePrompt,
#                     max_actions_per_step=1,
#                     use_vision=True,
#                     max_failures=1
#                 )
                
#                 step_result = await asyncio.wait_for(step_agent.run(max_steps=3), timeout=60.0)
#                 raw_output = step_result.final_result() or f"Completed step {step.id}"
                
#             except Exception as e:
#                 logger.error(f"❌ Step {step.id} failed: {e}")
#                 raw_output = f"Step failed: {str(e)}"
            
#             # PHASE 1: POST-STEP ANALYSIS (with goal context)
#             await send_socket_message_to_session(session, {
#                 "type": "thinking", 
#                 "data": {"step_id": step.id, "phase": "validating", "message": "Validating results against goal..."}
#             })
            
#             # Update current URL
#             try:
#                 current_page = await session.browser_context.get_current_page()
#                 session.execution_context['current_url'] = current_page.url
#             except:
#                 pass
            
#             result_analysis = await StepAnalyzer.analyze_after_step(step_dict, raw_output, session.execution_context, llm, session.goal_tracker)
            
#             # Check for stuck patterns
#             if result_analysis.get('stuck_pattern'):
#                 stuck_info = result_analysis['stuck_pattern']
#                 log_to_socket(f"🔄 Stuck pattern detected: {stuck_info.get('reason')}")
                
#                 await send_socket_message_to_session(session, {
#                     "type": "stuck_detected",
#                     "data": {
#                         "reason": stuck_info.get('reason'),
#                         "solutions": stuck_info.get('solutions', []),
#                         "recommended_action": stuck_info.get('recommended_action')
#                     }
#                 })
            
#             # PHASE 2: ADAPTIVE PLANNING (enhanced with goal awareness)
#             adaptation = None
#             if result_analysis.get('status') in ['failed', 'partial'] or result_analysis.get('stuck_pattern'):
#                 await send_socket_message_to_session(session, {
#                     "type": "thinking", 
#                     "data": {"step_id": step.id, "phase": "adapting", "message": "Planning goal-oriented adaptation..."}
#                 })
                
#                 remaining_steps = [{"id": s.id, "action": s.action, "description": s.description} for s in current_steps[i+1:]]
#                 adaptation = await AdaptivePlanner.should_adapt_plan(result_analysis, remaining_steps, llm)
                
#                 if adaptation:
#                     session.execution_context['adaptations_made'].append({
#                         "step_id": step.id,
#                         "reason": adaptation.get('reason'),
#                         "action": adaptation.get('action'),
#                         "goal_context": session.goal_tracker.main_goal,
#                         "timestamp": datetime.now().isoformat()
#                     })
#                     log_to_socket(f"🔄 Goal-oriented adaptation: {adaptation.get('reason')}")
            
#             # Update memory and context
#             step_memory = f"Step {step.id}: {result_analysis.get('what_happened', raw_output)}\n"
#             session.execution_context['memory_chain'] += step_memory
#             session.execution_context['variables'].update(result_analysis.get('data_extracted', {}))
            
#             # Update goal progress
#             progress_delta = result_analysis.get('goal_progress_delta', 0)
#             if progress_delta > 0:
#                 session.goal_tracker.current_progress = min(100, session.goal_tracker.current_progress + progress_delta)
            
#             # Store results
#             session.execution_context['step_results'].append({
#                 "step_id": step.id,
#                 "result": raw_output,
#                 "analysis": result_analysis,
#                 "adaptation": adaptation,
#                 "goal_progress": session.goal_tracker.current_progress,
#                 "completed_at": datetime.now().isoformat()
#             })
            
#             session.executed_steps.append({
#                 "step_id": step.id,
#                 "description": step.description,
#                 "result": raw_output,
#                 "analysis": result_analysis,
#                 "goal_contribution": step_analysis.get('goal_contribution'),
#                 "completed_at": datetime.now().isoformat()
#             })
            
#             # Send completion with goal tracking
#             await send_socket_message_to_session(session, {
#                 "type": "step_complete",
#                 "data": {
#                     "step_id": step.id,
#                     "step_number": i + 1,
#                     "result": raw_output,
#                     "analysis": result_analysis,
#                     "adaptation": adaptation,
#                     "goal_progress": session.goal_tracker.current_progress,
#                     "goal_summary": session.goal_tracker.get_goal_summary(),
#                     "variables": session.execution_context['variables']
#                 }
#             })
            
#             log_to_socket(f"✅ Step {step.id}: {result_analysis.get('status', 'completed')} (Goal: {session.goal_tracker.current_progress}%)")
            
#             # PHASE 2: APPLY ADAPTATIONS (same as before)
#             if adaptation:
#                 if adaptation.get('action') == 'retry':
#                     log_to_socket("🔄 Retrying step with goal focus")
#                     continue
#                 elif adaptation.get('action') == 'skip':
#                     log_to_socket("⏭️ Skipping to next step")
#                 elif adaptation.get('action') == 'add_step':
#                     new_step_data = adaptation.get('new_step', {})
#                     if new_step_data:
#                         class TempStep:
#                             def __init__(self, data):
#                                 self.id = data.get('id', step.id + 0.5)
#                                 self.action = data.get('action', 'wait')
#                                 self.description = data.get('description', 'Goal-oriented adaptive step')
#                                 self.params = data.get('params', {})
                        
#                         new_step = TempStep(new_step_data)
#                         current_steps.insert(i + 1, new_step)
#                         log_to_socket(f"➕ Added goal-focused step: {new_step.description}")
            
#             i += 1
        
#         # Final goal completion
#         final_goal_summary = session.goal_tracker.get_goal_summary()
        
#         await send_socket_message_to_session(session, {
#             "type": "steps_complete", 
#             "data": {
#                 "total_completed": len(current_steps),
#                 "adaptations_made": len(session.execution_context.get('adaptations_made', [])),
#                 "session_persistent": True,
#                 "goal_tracking_summary": {
#                     "main_goal": session.goal_tracker.main_goal,
#                     "final_progress": session.goal_tracker.current_progress,
#                     "deviation_count": session.goal_tracker.deviation_count,
#                     "sub_goals": session.goal_tracker.sub_goals,
#                     "goal_achieved": session.goal_tracker.current_progress >= 90
#                 },
#                 "intelligence_summary": {
#                     "successful_steps": len([r for r in session.execution_context['step_results'] if r.get('analysis', {}).get('status') == 'success']),
#                     "adaptations": session.execution_context.get('adaptations_made', []),
#                     "variables": session.execution_context['variables']
#                 }
#             }
#         })
        
#         goal_status = "🎯 GOAL ACHIEVED!" if session.goal_tracker.current_progress >= 90 else f"🎯 Goal {session.goal_tracker.current_progress}% complete"
       
#         log_to_socket(f"{goal_status} - Made {len(session.execution_context.get('adaptations_made', []))} adaptations, {session.goal_tracker.deviation_count} course corrections")
       
#     except Exception as e:
#         logger.error(f"❌ Goal-oriented execution failed: {e}", exc_info=True)
#         log_to_socket(f"❌ Execution failed: {str(e)}")
#         await send_socket_message_to_session(session, {"type": "error", "data": str(e)})

async def run_steps_logic(config: ExecuteStepsRequest, session: UserSession):
    """Execute steps with intelligence, adaptation, goal tracking AND advanced features"""  # 🆕 NEW: Updated description
    logger.info(f"🚀 Starting goal-oriented execution for session {session.session_id}")
    
    # Import intelligence modules
    from src.intelligence.step_analyzer import StepAnalyzer, AdaptivePlanner
    from src.intelligence.goal_tracker import GoalTracker
    
    # Initialize session state safely
    if not hasattr(session, 'execution_context') or session.execution_context is None:
        session.execution_context = {}
    
    session.execution_context.setdefault('current_url', '')
    session.execution_context.setdefault('variables', {})
    session.execution_context.setdefault('step_results', [])
    session.execution_context.setdefault('memory_chain', '')
    session.execution_context.setdefault('adaptations_made', [])
    
    # 🆕 NEW: Enhanced context setup for advanced features
    session.execution_context.setdefault('advanced_features', {
        'generate_analysis': getattr(config, 'generate_analysis', False),
        'export_formats': getattr(config, 'export_formats', []),
        'popup_killer': getattr(config, 'enable_popup_killer', True),
        'captcha_avoidance': getattr(config, 'enable_captcha_avoidance', True),
        'force_duckduckgo': getattr(config, 'force_duckduckgo', True)
    })
    
    if not hasattr(session, 'executed_steps'):
        session.executed_steps = []
    if not hasattr(session, 'current_step_index'):
        session.current_step_index = 0
    
    def log_to_socket(msg, *args, **kwargs):
        log_message = msg % args if args else str(msg)
        asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": log_message}))
    
    try:
        # Initialize LLM
        llm = await _initialize_llm_helper(config.llm_provider, config.llm_model_name, config.llm_api_key)
        log_to_socket("🧠 Intelligent agent with goal tracking initialized")
        
        # 🆕 NEW: Show enabled features
        advanced_features = session.execution_context['advanced_features']
        if advanced_features.get('popup_killer'):
            log_to_socket("🛡️ Auto popup killer enabled")
        if advanced_features.get('force_duckduckgo'):
            log_to_socket("🔒 Privacy mode: All searches will use DuckDuckGo only")
        if advanced_features.get('captcha_avoidance'):
            log_to_socket("🤖 Captcha avoidance enabled")
        if advanced_features.get('generate_analysis'):
            log_to_socket("📊 Analysis report will be generated")
        if advanced_features.get('export_formats'):
            log_to_socket(f"📁 Data export formats: {', '.join(advanced_features['export_formats'])}")
        
        # Create browser if needed
        if not session.browser:
            log_to_socket("🌐 Creating browser...")
            session.browser = CustomBrowser(config=BrowserConfig(headless=True))
            session.browser_context = await session.browser.new_context(
                config=BrowserContextConfig(
                    no_viewport=False, 
                    browser_window_size=BrowserContextWindowSize(width=1280, height=720),
                    save_downloads_path=f"/tmp/downloads_{session.session_id}"
                )
            )
        
        # Initialize Goal Tracker
        if not hasattr(session, 'goal_tracker') or session.goal_tracker is None:
            # Extract main goal from first step or overall context
            main_goal = f"Complete task: {config.steps[0].description if config.steps else 'Execute steps'}"
            session.goal_tracker = GoalTracker(main_goal)
            
            # Decompose goal into sub-goals
            await session.goal_tracker.decompose_goal(llm)
            log_to_socket(f"🎯 Goal set: {main_goal}")
            
            # Send goal information to frontend
            await send_socket_message_to_session(session, {
                "type": "goal_initialized",
                "data": {
                    "main_goal": main_goal,
                    "sub_goals": session.goal_tracker.sub_goals,
                    "progress": 0
                }
            })
        
        # Convert steps to mutable list for adaptation
        current_steps = list(config.steps[config.start_from:])
        i = 0
        
        while i < len(current_steps):
            step = current_steps[i]
            step_dict = {"id": step.id, "action": step.action, "description": step.description, "params": step.params}
            
            # 🆕 NEW: Smart page preparation - FIRST STEP ONLY
            if i == 0 and session.execution_context['advanced_features'].get('popup_killer', True):
                log_to_socket("🛡️ Preparing page - killing popups and checking for issues...")
                
                try:
                    # Get current page
                    current_page = await session.browser_context.get_current_page()
                    
                    # Auto popup killer using JavaScript
                    prep_result = await current_page.evaluate("""
                        () => {
                            try {
                                // Send ESC key
                                document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
                                
                                // Find and close common popups
                                const selectors = [
                                    '.modal', '.popup', '.overlay', '.lightbox',
                                    '[class*="modal"]', '[class*="popup"]', '[class*="overlay"]',
                                    '[class*="cookie"]', '[class*="gdpr"]', '[class*="consent"]',
                                    '.close', '.close-btn', '.close-button',
                                    'button[aria-label*="close" i]', 'button[aria-label*="dismiss" i]'
                                ];
                                
                                let closed = 0;
                                for (const sel of selectors) {
                                    const elements = document.querySelectorAll(sel);
                                    elements.forEach(el => {
                                        if (el.offsetParent !== null && el.style.display !== 'none') {
                                            try {
                                                el.click();
                                                closed++;
                                            } catch(e) {
                                                // Ignore click errors
                                            }
                                        }
                                    });
                                }
                                
                                // Additional ESC for good measure
                                document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
                                
                                return `Popup killer: ${closed} elements handled`;
                            } catch(e) {
                                return `Popup killer error: ${e.message}`;
                            }
                        }
                    """)
                    
                    log_to_socket(f"🛡️ {prep_result}")
                    await asyncio.sleep(1)  # Brief pause after popup cleanup
                    
                except Exception as e:
                    log_to_socket(f"🛡️ Popup killer error: {str(e)}")
            
            log_to_socket(f"🎯 Step {i+1}/{len(current_steps)}: {step.description}")
            
            # GOAL TRACKING: Check alignment before step
            await send_socket_message_to_session(session, {
                "type": "thinking", 
                "data": {"step_id": step.id, "phase": "goal_checking", "message": "Checking goal alignment..."}
            })
            
            alignment = await session.goal_tracker.check_alignment(
                session.execution_context, 
                step.description, 
                llm
            )
            
            # Send goal status update
            await send_socket_message_to_session(session, {
                "type": "goal_status",
                "data": {
                    "alignment": alignment.get('alignment'),
                    "progress": alignment.get('progress_percentage', 0),
                    "reasoning": alignment.get('reasoning', ''),
                    "next_focus": alignment.get('next_focus', '')
                }
            })
            
            # Handle major deviations
            if alignment.get('alignment') == 'major_deviation':
                log_to_socket(f"⚠️ Major deviation detected: {alignment.get('reasoning')}")
                
                correction = await session.goal_tracker.suggest_course_correction(
                    alignment.get('reasoning', 'Unknown deviation'),
                    session.execution_context,
                    llm
                )
                
                await send_socket_message_to_session(session, {
                    "type": "course_correction",
                    "data": {
                        "deviation_reason": alignment.get('reasoning'),
                        "correction": correction,
                        "urgency": correction.get('urgency', 'medium')
                    }
                })
                
                log_to_socket(f"🔄 Course correction: {correction.get('action')}")
                
                # Apply correction if needed
                if correction.get('correction_type') == 'skip_step':
                    log_to_socket("⏭️ Skipping step due to course correction")
                    i += 1
                    continue
            
            # 🆕 NEW: Captcha avoidance check
            # if session.execution_context['advanced_features'].get('captcha_avoidance', True):
            #     try:
            #         current_page = await session.browser_context.get_current_page()
            #         captcha_check = await current_page.evaluate("""
            #             () => {
            #                 const captchaSelectors = [
            #                     '.captcha', '.recaptcha', '.hcaptcha',
            #                     '[id*="captcha"]', '[class*="captcha"]',
            #                     'iframe[src*="recaptcha"]', 'iframe[src*="hcaptcha"]',
            #                     '.g-recaptcha', '#recaptcha'
            #                 ];
                            
            #                 for (const sel of captchaSelectors) {
            #                     const el = document.querySelector(sel);
            #                     if (el && el.offsetParent !== null) {
            #                         return 'captcha_detected';
            #                     }
            #                 }
            #                 return 'no_captcha';
            #             }
            #         """)
                    
            #         if captcha_check == 'captcha_detected':
            #             log_to_socket("🤖 Captcha detected - attempting avoidance strategies")
                        
            #             await send_socket_message_to_session(session, {
            #                 "type": "captcha_detected",
            #                 "data": {
            #                     "message": "Captcha detected, implementing avoidance strategy",
            #                     "strategies": ["wait_retry", "alternative_approach"]
            #                 }
            #             })
                        
            #             # Simple avoidance: wait and retry
            #             await asyncio.sleep(3)
            #             await current_page.reload()
            #             log_to_socket("🔄 Page reloaded to avoid captcha")
                        
            #     except Exception as e:
            #         log_to_socket(f"🤖 Captcha check error: {str(e)}")
            
            # PHASE 1: PRE-STEP ANALYSIS (with goal context)
            await send_socket_message_to_session(session, {
                "type": "thinking", 
                "data": {"step_id": step.id, "phase": "analyzing", "message": "Analyzing step with goal context..."}
            })
            
            step_analysis = await StepAnalyzer.analyze_before_step(step_dict, session.execution_context, llm, session.goal_tracker)
            log_to_socket(f"🧠 Goal: {step_analysis.get('goal', 'Execute step')}")
            
            # Send step start with analysis
            await send_socket_message_to_session(session, {
                "type": "step_start", 
                "data": {
                    "step_id": step.id,
                    "step_number": i + 1,
                    "total": len(current_steps),
                    "description": step.description,
                    "analysis": step_analysis,
                    "goal_alignment": step_analysis.get('goal_alignment')
                }
            })
            
            # EXECUTE STEP
            try:
                # 🆕 NEW: Enhanced context prompt with all features
                duckduckgo_instruction = ""
                if session.execution_context['advanced_features'].get('force_duckduckgo', True):
                    duckduckgo_instruction = "\nPRIVACY MODE: Use only DuckDuckGo for searches, never Google or other search engines."
                
                context_prompt = f"""You are an intelligent browser agent working toward a specific goal.

MAIN GOAL: {session.goal_tracker.main_goal}
GOAL PROGRESS: {session.goal_tracker.current_progress}%

CURRENT STEP:
GOAL: {step_analysis.get('goal', step.description)}
ACTION: {step.action}
PARAMETERS: {json.dumps(step.params)}
GOAL CONTRIBUTION: {step_analysis.get('goal_contribution', 'Unknown')}

CONTEXT:
{session.execution_context.get('memory_chain', '')[-400:]}

EXPECTATIONS: {step_analysis.get('expectations', 'Unknown')}
SUCCESS CRITERIA: {step_analysis.get('success_criteria', 'Step completes')}
{duckduckgo_instruction}

IMPORTANT: Stay focused on the main goal. If you encounter distractions (popups, ads, irrelevant content), ignore them and stay on track toward: {session.goal_tracker.main_goal}

Execute this step intelligently while keeping the main goal in mind."""
                
                await send_socket_message_to_session(session, {
                    "type": "thinking", 
                    "data": {"step_id": step.id, "phase": "executing", "message": f"Executing: {step_analysis.get('goal', step.description)}"}
                })
                
                # Create and run step agent
                step_agent = CustomAgent(
                    task=context_prompt,
                    llm=llm,
                    browser=session.browser,
                    browser_context=session.browser_context,
                    controller=CustomController(),
                    system_prompt_class=CustomSystemPrompt,
                    agent_prompt_class=CustomAgentMessagePrompt,
                    max_actions_per_step=1,
                    use_vision=True,
                    max_failures=1
                )
                
                step_result = await asyncio.wait_for(step_agent.run(max_steps=3), timeout=60.0)
                raw_output = step_result.final_result() or f"Completed step {step.id}"
                
            except Exception as e:
                logger.error(f"❌ Step {step.id} failed: {e}")
                raw_output = f"Step failed: {str(e)}"
            
            # PHASE 1: POST-STEP ANALYSIS (with goal context)
            await send_socket_message_to_session(session, {
                "type": "thinking", 
                "data": {"step_id": step.id, "phase": "validating", "message": "Validating results against goal..."}
            })
            
            # Update current URL
            try:
                current_page = await session.browser_context.get_current_page()
                session.execution_context['current_url'] = current_page.url
            except:
                pass
            
            result_analysis = await StepAnalyzer.analyze_after_step(step_dict, raw_output, session.execution_context, llm, session.goal_tracker)
            
            # Check for stuck patterns
            if result_analysis.get('stuck_pattern'):
                stuck_info = result_analysis['stuck_pattern']
                log_to_socket(f"🔄 Stuck pattern detected: {stuck_info.get('reason')}")
                
                await send_socket_message_to_session(session, {
                    "type": "stuck_detected",
                    "data": {
                        "reason": stuck_info.get('reason'),
                        "solutions": stuck_info.get('solutions', []),
                        "recommended_action": stuck_info.get('recommended_action')
                    }
                })
            
            # PHASE 2: ADAPTIVE PLANNING (enhanced with goal awareness)
            adaptation = None
            if result_analysis.get('status') in ['failed', 'partial'] or result_analysis.get('stuck_pattern'):
                await send_socket_message_to_session(session, {
                    "type": "thinking", 
                    "data": {"step_id": step.id, "phase": "adapting", "message": "Planning goal-oriented adaptation..."}
                })
                
                remaining_steps = [{"id": s.id, "action": s.action, "description": s.description} for s in current_steps[i+1:]]
                adaptation = await AdaptivePlanner.should_adapt_plan(result_analysis, remaining_steps, llm)
                
                if adaptation:
                    session.execution_context['adaptations_made'].append({
                        "step_id": step.id,
                        "reason": adaptation.get('reason'),
                        "action": adaptation.get('action'),
                        "goal_context": session.goal_tracker.main_goal,
                        "timestamp": datetime.now().isoformat()
                    })
                    log_to_socket(f"🔄 Goal-oriented adaptation: {adaptation.get('reason')}")
            
            # Update memory and context
            step_memory = f"Step {step.id}: {result_analysis.get('what_happened', raw_output)}\n"
            session.execution_context['memory_chain'] += step_memory
            session.execution_context['variables'].update(result_analysis.get('data_extracted', {}))
            
            # Update goal progress
            progress_delta = result_analysis.get('goal_progress_delta', 0)
            if progress_delta > 0:
                session.goal_tracker.current_progress = min(100, session.goal_tracker.current_progress + progress_delta)
            
            # Store results
            session.execution_context['step_results'].append({
                "step_id": step.id,
                "result": raw_output,
                "analysis": result_analysis,
                "adaptation": adaptation,
                "goal_progress": session.goal_tracker.current_progress,
                "completed_at": datetime.now().isoformat()
            })
            
            session.executed_steps.append({
                "step_id": step.id,
                "description": step.description,
                "result": raw_output,
                "analysis": result_analysis,
                "goal_contribution": step_analysis.get('goal_contribution'),
                "completed_at": datetime.now().isoformat()
            })
            
            # Send completion with goal tracking
            await send_socket_message_to_session(session, {
                "type": "step_complete",
                "data": {
                    "step_id": step.id,
                    "step_number": i + 1,
                    "result": raw_output,
                    "analysis": result_analysis,
                    "adaptation": adaptation,
                    "goal_progress": session.goal_tracker.current_progress,
                    "goal_summary": session.goal_tracker.get_goal_summary(),
                    "variables": session.execution_context['variables']
                }
            })
            
            log_to_socket(f"✅ Step {step.id}: {result_analysis.get('status', 'completed')} (Goal: {session.goal_tracker.current_progress}%)")
            
            # PHASE 2: APPLY ADAPTATIONS (same as before)
            if adaptation:
                if adaptation.get('action') == 'retry':
                    log_to_socket("🔄 Retrying step with goal focus")
                    continue
                elif adaptation.get('action') == 'skip':
                    log_to_socket("⏭️ Skipping to next step")
                elif adaptation.get('action') == 'add_step':
                    new_step_data = adaptation.get('new_step', {})
                    if new_step_data:
                        class TempStep:
                            def __init__(self, data):
                                self.id = data.get('id', step.id + 0.5)
                                self.action = data.get('action', 'wait')
                                self.description = data.get('description', 'Goal-oriented adaptive step')
                                self.params = data.get('params', {})
                        
                        new_step = TempStep(new_step_data)
                        current_steps.insert(i + 1, new_step)
                        log_to_socket(f"➕ Added goal-focused step: {new_step.description}")
            
            i += 1
        
        # Final goal completion
        final_goal_summary = session.goal_tracker.get_goal_summary()
        
        await send_socket_message_to_session(session, {
            "type": "steps_complete", 
            "data": {
                "total_completed": len(current_steps),
                "adaptations_made": len(session.execution_context.get('adaptations_made', [])),
                "session_persistent": True,
                "goal_tracking_summary": {
                    "main_goal": session.goal_tracker.main_goal,
                    "final_progress": session.goal_tracker.current_progress,
                    "deviation_count": session.goal_tracker.deviation_count,
                    "sub_goals": session.goal_tracker.sub_goals,
                    "goal_achieved": session.goal_tracker.current_progress >= 90
                },
                "intelligence_summary": {
                    "successful_steps": len([r for r in session.execution_context['step_results'] if r.get('analysis', {}).get('status') == 'success']),
                    "adaptations": session.execution_context.get('adaptations_made', []),
                    "variables": session.execution_context['variables']
                }
            }
        })
        
        goal_status = "🎯 GOAL ACHIEVED!" if session.goal_tracker.current_progress >= 90 else f"🎯 Goal {session.goal_tracker.current_progress}% complete"
       
        # 🆕 NEW: Generate analysis report and exports if requested
        advanced_features = session.execution_context['advanced_features']
        if advanced_features.get('generate_analysis') or advanced_features.get('export_formats'):
            log_to_socket("📊 Generating analysis report and exports...")
            
            try:
                from src.intelligence.report_generator import ReportGenerator
                
                report = await ReportGenerator.generate_report_with_exports(
                    {"execution_context": session.execution_context, "goal_tracker": session.goal_tracker},
                    llm,
                    advanced_features.get('export_formats', [])
                )
                
                # Send report to frontend
                await send_socket_message_to_session(session, {
                    "type": "analysis_report",
                    "data": report
                })
                
                log_to_socket("✅ Analysis report and exports generated successfully")
                
            except Exception as e:
                log_to_socket(f"❌ Analysis generation failed: {e}")
       
        log_to_socket(f"{goal_status} - Made {len(session.execution_context.get('adaptations_made', []))} adaptations, {session.goal_tracker.deviation_count} course corrections")
       
    except Exception as e:
        logger.error(f"❌ Goal-oriented execution failed: {e}", exc_info=True)
        log_to_socket(f"❌ Execution failed: {str(e)}")
        await send_socket_message_to_session(session, {"type": "error", "data": str(e)})

async def _extract_structured_result(step: Step, raw_output: str, browser_context, session_id: str):
    """Extract structured data from step execution with session tracking"""
    import re
    try:
        current_page = await browser_context.get_current_page()
        current_url = current_page.url
        page_title = await current_page.title()
        
        # Base result structure
        result = {
            "step_id": step.id,
            "action": step.action,
            "session_id": session_id,
            "status": "success" if all(keyword not in raw_output.lower() 
                                    for keyword in ["error", "failed", "timeout"]) else "failed",
            "raw_output": raw_output[:500] + "..." if len(raw_output) > 500 else raw_output,
            "url_before": current_url,
            "url_after": current_url,
            "page_title": page_title,
            "extracted_variables": {},
            "next_step_context": "",
            "execution_metadata": {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
        }
        
        # Action-specific data extraction
        if step.action in ["search", "find", "go_to_url"]:
            # Navigation/Search actions
            if any(keyword in raw_output.lower() for keyword in ["found", "navigated", "loaded"]):
                result["extracted_variables"]["navigation_success"] = True
                result["next_step_context"] = f"Successfully navigated to {current_url}"
            
            # Extract search terms or URLs
            if "search" in step.action and step.params.get("query"):
                result["extracted_variables"]["search_query"] = step.params["query"]
                result["next_step_context"] = f"Search completed for: {step.params['query']}"
                
        elif step.action in ["extract", "get_price", "get_text", "extract_content"]:
            # Data extraction actions
            
            
            # Extract prices
            price_patterns = [
                r'\$[\d,]+\.?\d*',
                r'USD\s*[\d,]+\.?\d*',
                r'Price:?\s*\$?[\d,]+\.?\d*'
            ]
            
            for pattern in price_patterns:
                price_match = re.search(pattern, raw_output, re.IGNORECASE)
                if price_match:
                    result["extracted_variables"]["price"] = price_match.group().strip()
                    break
            
            # Extract product information
            if any(keyword in raw_output.lower() for keyword in ["product", "item", "title"]):
                result["extracted_variables"]["product_info_found"] = True
                
            # Extract any quoted text (likely extracted content)
            quoted_text = re.findall(r'"([^"]*)"', raw_output)
            if quoted_text:
                result["extracted_variables"]["extracted_text"] = quoted_text[0][:200]
                
        elif step.action in ["click", "add_to_cart", "submit", "button_click"]:
            # Action/Interaction steps
            success_indicators = ["clicked", "added", "submitted", "successful", "completed"]
            if any(indicator in raw_output.lower() for indicator in success_indicators):
                result["extracted_variables"]["action_completed"] = True
                result["next_step_context"] = f"Action '{step.action}' completed successfully"
            else:
                result["status"] = "uncertain"
                result["next_step_context"] = f"Action '{step.action}' attempted, verify results"
        
        # Extract any URLs mentioned in output
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls_found = re.findall(url_pattern, raw_output)
        if urls_found:
            result["extracted_variables"]["urls_found"] = urls_found[:3]  # Limit to first 3
        
        # Check for error indicators
        error_keywords = ["error", "failed", "timeout", "not found", "unable"]
        if any(keyword in raw_output.lower() for keyword in error_keywords):
            result["status"] = "failed"
            result["extracted_variables"]["error_detected"] = True
        
        return result
        
    except Exception as e:
        return {
            "step_id": step.id,
            "action": step.action,
            "session_id": session_id,
            "status": "error",
            "raw_output": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output,
            "error": str(e),
            "extracted_variables": {},
            "next_step_context": f"Step {step.id} encountered an error",
            "execution_metadata": {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "error": True
            }
        }

async def _build_context_prompt(step: Step, session: UserSession) -> str:
    """Build context-aware prompt for step execution with session isolation"""
    
    session_context = session.execution_context
    recent_results = session_context.get('step_results', [])[-2:]  # Last 2 steps
    
    context_prompt = f"""
EXECUTE THIS STEP WITH FULL SESSION CONTEXT:

SESSION INFO:
- Session ID: {session.session_id}
- Current URL: {session_context.get('current_url', 'Not set')}
- Browser Ready: {session_context.get('browser_ready', False)}

CURRENT STEP:
- Step ID: {step.id}
- Action: {step.action}
- Description: {step.description}
- Parameters: {json.dumps(step.params, indent=2)}

EXECUTION CONTEXT:
- Available Variables: {json.dumps(session_context.get('variables', {}), indent=2)}
- Last Action: {session_context.get('last_action', 'None')}
- Steps Completed: {len(session.executed_steps)}

RECENT STEP RESULTS:
{json.dumps(recent_results, indent=2) if recent_results else "No previous steps in this session"}

INSTRUCTIONS:
1. Execute ONLY the specified action for this step
2. Use the session context and variables from previous steps
3. If referencing previous results, use the variables provided
4. Extract any important data for subsequent steps
5. Be precise and focused on this specific step

Remember: This is session {session.session_id[:8]} - maintain context isolation from other users.
"""
    
    return context_prompt



# async def run_steps_logic(config: ExecuteStepsRequest, session: UserSession):
#     """Execute steps with intelligence, adaptation, and goal tracking"""
#     logger.info(f"🚀 Starting goal-oriented execution for session {session.session_id}")
    
#     # Import intelligence modules
#     from src.intelligence.step_analyzer import StepAnalyzer, AdaptivePlanner
#     from src.intelligence.goal_tracker import GoalTracker
    
#     # Initialize session state safely
#     if not hasattr(session, 'execution_context') or session.execution_context is None:
#         session.execution_context = {}
    
#     session.execution_context.setdefault('current_url', '')
#     session.execution_context.setdefault('variables', {})
#     session.execution_context.setdefault('step_results', [])
#     session.execution_context.setdefault('memory_chain', '')
#     session.execution_context.setdefault('adaptations_made', [])
    
#     if not hasattr(session, 'executed_steps'):
#         session.executed_steps = []
#     if not hasattr(session, 'current_step_index'):
#         session.current_step_index = 0
    
#     def log_to_socket(msg, *args, **kwargs):
#         log_message = msg % args if args else str(msg)
#         asyncio.create_task(send_socket_message_to_session(session, {"type": "log", "data": log_message}))
    
#     try:
#         # Initialize LLM
#         llm = await _initialize_llm_helper(config.llm_provider, config.llm_model_name, config.llm_api_key)
#         log_to_socket("🧠 Intelligent agent with goal tracking initialized")
        
#         # Create browser if needed
#         if not session.browser:
#             log_to_socket("🌐 Creating browser...")
#             session.browser = CustomBrowser(config=BrowserConfig(headless=False))
#             session.browser_context = await session.browser.new_context(
#                 config=BrowserContextConfig(
#                     no_viewport=False, 
#                     browser_window_size=BrowserContextWindowSize(width=1280, height=720),
#                     save_downloads_path=f"/tmp/downloads_{session.session_id}"
#                 )
#             )
        
#         # Initialize Goal Tracker
#         if not hasattr(session, 'goal_tracker') or session.goal_tracker is None:
#             # Extract main goal from first step or overall context
#             main_goal = f"Complete task: {config.steps[0].description if config.steps else 'Execute steps'}"
#             session.goal_tracker = GoalTracker(main_goal)
            
#             # Decompose goal into sub-goals
#             await session.goal_tracker.decompose_goal(llm)
#             log_to_socket(f"🎯 Goal set: {main_goal}")
            
#             # Send goal information to frontend
#             await send_socket_message_to_session(session, {
#                 "type": "goal_initialized",
#                 "data": {
#                     "main_goal": main_goal,
#                     "sub_goals": session.goal_tracker.sub_goals,
#                     "progress": 0
#                 }
#             })
        
#         # Convert steps to mutable list for adaptation
#         current_steps = list(config.steps[config.start_from:])
#         i = 0
        
#         while i < len(current_steps):
#             step = current_steps[i]
#             step_dict = {"id": step.id, "action": step.action, "description": step.description, "params": step.params}
            
#             log_to_socket(f"🎯 Step {i+1}/{len(current_steps)}: {step.description}")
            
#             # GOAL TRACKING: Check alignment before step
#             await send_socket_message_to_session(session, {
#                 "type": "thinking", 
#                 "data": {"step_id": step.id, "phase": "goal_checking", "message": "Checking goal alignment..."}
#             })
            
#             alignment = await session.goal_tracker.check_alignment(
#                 session.execution_context, 
#                 step.description, 
#                 llm
#             )
            
#             # Send goal status update
#             await send_socket_message_to_session(session, {
#                 "type": "goal_status",
#                 "data": {
#                     "alignment": alignment.get('alignment'),
#                     "progress": alignment.get('progress_percentage', 0),
#                     "reasoning": alignment.get('reasoning', ''),
#                     "next_focus": alignment.get('next_focus', '')
#                 }
#             })
            
#             # Handle major deviations
#             if alignment.get('alignment') == 'major_deviation':
#                 log_to_socket(f"⚠️ Major deviation detected: {alignment.get('reasoning')}")
                
#                 correction = await session.goal_tracker.suggest_course_correction(
#                     alignment.get('reasoning', 'Unknown deviation'),
#                     session.execution_context,
#                     llm
#                 )
                
#                 await send_socket_message_to_session(session, {
#                     "type": "course_correction",
#                     "data": {
#                         "deviation_reason": alignment.get('reasoning'),
#                         "correction": correction,
#                         "urgency": correction.get('urgency', 'medium')
#                     }
#                 })
                
#                 log_to_socket(f"🔄 Course correction: {correction.get('action')}")
                
#                 # Apply correction if needed
#                 if correction.get('correction_type') == 'skip_step':
#                     log_to_socket("⏭️ Skipping step due to course correction")
#                     i += 1
#                     continue
            
#             # PHASE 1: PRE-STEP ANALYSIS (with goal context)
#             await send_socket_message_to_session(session, {
#                 "type": "thinking", 
#                 "data": {"step_id": step.id, "phase": "analyzing", "message": "Analyzing step with goal context..."}
#             })
            
#             step_analysis = await StepAnalyzer.analyze_before_step(step_dict, session.execution_context, llm, session.goal_tracker)
#             log_to_socket(f"🧠 Goal: {step_analysis.get('goal', 'Execute step')}")
            
#             # Send step start with analysis
#             await send_socket_message_to_session(session, {
#                 "type": "step_start", 
#                 "data": {
#                     "step_id": step.id,
#                     "step_number": i + 1,
#                     "total": len(current_steps),
#                     "description": step.description,
#                     "analysis": step_analysis,
#                     "goal_alignment": step_analysis.get('goal_alignment')
#                 }
#             })
            
#             # EXECUTE STEP
#             try:
#                 # Build intelligent context prompt with goal awareness
#                 context_prompt = f"""You are an intelligent browser agent working toward a specific goal.

# MAIN GOAL: {session.goal_tracker.main_goal}
# GOAL PROGRESS: {session.goal_tracker.current_progress}%

# CURRENT STEP:
# GOAL: {step_analysis.get('goal', step.description)}
# ACTION: {step.action}
# PARAMETERS: {json.dumps(step.params)}
# GOAL CONTRIBUTION: {step_analysis.get('goal_contribution', 'Unknown')}

# CONTEXT:
# {session.execution_context.get('memory_chain', '')[-400:]}

# EXPECTATIONS: {step_analysis.get('expectations', 'Unknown')}
# SUCCESS CRITERIA: {step_analysis.get('success_criteria', 'Step completes')}

# IMPORTANT: Stay focused on the main goal. If you encounter distractions (popups, ads, irrelevant content), ignore them and stay on track toward: {session.goal_tracker.main_goal}

# Execute this step intelligently while keeping the main goal in mind."""
                
#                 await send_socket_message_to_session(session, {
#                     "type": "thinking", 
#                     "data": {"step_id": step.id, "phase": "executing", "message": f"Executing: {step_analysis.get('goal', step.description)}"}
#                 })
                
#                 # Create and run step agent
#                 step_agent = CustomAgent(
#                     task=context_prompt,
#                     llm=llm,
#                     browser=session.browser,
#                     browser_context=session.browser_context,
#                     controller=CustomController(),
#                     system_prompt_class=CustomSystemPrompt,
#                     agent_prompt_class=CustomAgentMessagePrompt,
#                     max_actions_per_step=1,
#                     use_vision=True,
#                     max_failures=1
#                 )
                
#                 step_result = await asyncio.wait_for(step_agent.run(max_steps=3), timeout=60.0)
#                 raw_output = step_result.final_result() or f"Completed step {step.id}"
                
#             except Exception as e:
#                 logger.error(f"❌ Step {step.id} failed: {e}")
#                 raw_output = f"Step failed: {str(e)}"
            
#             # PHASE 1: POST-STEP ANALYSIS (with goal context)
#             await send_socket_message_to_session(session, {
#                 "type": "thinking", 
#                 "data": {"step_id": step.id, "phase": "validating", "message": "Validating results against goal..."}
#             })
            
#             # Update current URL
#             try:
#                 current_page = await session.browser_context.get_current_page()
#                 session.execution_context['current_url'] = current_page.url
#             except:
#                 pass
            
#             result_analysis = await StepAnalyzer.analyze_after_step(step_dict, raw_output, session.execution_context, llm, session.goal_tracker)
            
#             # Check for stuck patterns
#             if result_analysis.get('stuck_pattern'):
#                 stuck_info = result_analysis['stuck_pattern']
#                 log_to_socket(f"🔄 Stuck pattern detected: {stuck_info.get('reason')}")
                
#                 await send_socket_message_to_session(session, {
#                     "type": "stuck_detected",
#                     "data": {
#                         "reason": stuck_info.get('reason'),
#                         "solutions": stuck_info.get('solutions', []),
#                         "recommended_action": stuck_info.get('recommended_action')
#                     }
#                 })
            
#             # PHASE 2: ADAPTIVE PLANNING (enhanced with goal awareness)
#             adaptation = None
#             if result_analysis.get('status') in ['failed', 'partial'] or result_analysis.get('stuck_pattern'):
#                 await send_socket_message_to_session(session, {
#                     "type": "thinking", 
#                     "data": {"step_id": step.id, "phase": "adapting", "message": "Planning goal-oriented adaptation..."}
#                 })
                
#                 remaining_steps = [{"id": s.id, "action": s.action, "description": s.description} for s in current_steps[i+1:]]
#                 adaptation = await AdaptivePlanner.should_adapt_plan(result_analysis, remaining_steps, llm)
                
#                 if adaptation:
#                     session.execution_context['adaptations_made'].append({
#                         "step_id": step.id,
#                         "reason": adaptation.get('reason'),
#                         "action": adaptation.get('action'),
#                         "goal_context": session.goal_tracker.main_goal,
#                         "timestamp": datetime.now().isoformat()
#                     })
#                     log_to_socket(f"🔄 Goal-oriented adaptation: {adaptation.get('reason')}")
            
#             # Update memory and context
#             step_memory = f"Step {step.id}: {result_analysis.get('what_happened', raw_output)}\n"
#             session.execution_context['memory_chain'] += step_memory
#             session.execution_context['variables'].update(result_analysis.get('data_extracted', {}))
            
#             # Update goal progress
#             progress_delta = result_analysis.get('goal_progress_delta', 0)
#             if progress_delta > 0:
#                 session.goal_tracker.current_progress = min(100, session.goal_tracker.current_progress + progress_delta)
            
#             # Store results
#             session.execution_context['step_results'].append({
#                 "step_id": step.id,
#                 "result": raw_output,
#                 "analysis": result_analysis,
#                 "adaptation": adaptation,
#                 "goal_progress": session.goal_tracker.current_progress,
#                 "completed_at": datetime.now().isoformat()
#             })
            
#             session.executed_steps.append({
#                 "step_id": step.id,
#                 "description": step.description,
#                 "result": raw_output,
#                 "analysis": result_analysis,
#                 "goal_contribution": step_analysis.get('goal_contribution'),
#                 "completed_at": datetime.now().isoformat()
#             })
            
#             # Send completion with goal tracking
#             await send_socket_message_to_session(session, {
#                 "type": "step_complete",
#                 "data": {
#                     "step_id": step.id,
#                     "step_number": i + 1,
#                     "result": raw_output,
#                     "analysis": result_analysis,
#                     "adaptation": adaptation,
#                     "goal_progress": session.goal_tracker.current_progress,
#                     "goal_summary": session.goal_tracker.get_goal_summary(),
#                     "variables": session.execution_context['variables']
#                 }
#             })
            
#             log_to_socket(f"✅ Step {step.id}: {result_analysis.get('status', 'completed')} (Goal: {session.goal_tracker.current_progress}%)")
            
#             # PHASE 2: APPLY ADAPTATIONS (same as before)
#             if adaptation:
#                 if adaptation.get('action') == 'retry':
#                     log_to_socket("🔄 Retrying step with goal focus")
#                     continue
#                 elif adaptation.get('action') == 'skip':
#                     log_to_socket("⏭️ Skipping to next step")
#                 elif adaptation.get('action') == 'add_step':
#                     new_step_data = adaptation.get('new_step', {})
#                     if new_step_data:
#                         class TempStep:
#                             def __init__(self, data):
#                                 self.id = data.get('id', step.id + 0.5)
#                                 self.action = data.get('action', 'wait')
#                                 self.description = data.get('description', 'Goal-oriented adaptive step')
#                                 self.params = data.get('params', {})
                        
#                         new_step = TempStep(new_step_data)
#                         current_steps.insert(i + 1, new_step)
#                         log_to_socket(f"➕ Added goal-focused step: {new_step.description}")
            
#             i += 1
        
#         # Final goal completion
#         final_goal_summary = session.goal_tracker.get_goal_summary()
        
#         await send_socket_message_to_session(session, {
#             "type": "steps_complete", 
#             "data": {
#                 "total_completed": len(current_steps),
#                 "adaptations_made": len(session.execution_context.get('adaptations_made', [])),
#                 "session_persistent": True,
#                 "goal_tracking_summary": {
#                     "main_goal": session.goal_tracker.main_goal,
#                     "final_progress": session.goal_tracker.current_progress,
#                     "deviation_count": session.goal_tracker.deviation_count,
#                     "sub_goals": session.goal_tracker.sub_goals,
#                     "goal_achieved": session.goal_tracker.current_progress >= 90
#                 },
#                 "intelligence_summary": {
#                     "successful_steps": len([r for r in session.execution_context['step_results'] if r.get('analysis', {}).get('status') == 'success']),
#                     "adaptations": session.execution_context.get('adaptations_made', []),
#                     "variables": session.execution_context['variables']
#                 }
#             }
#         })
        
#         goal_status = "🎯 GOAL ACHIEVED!" if session.goal_tracker.current_progress >= 90 else f"🎯 Goal {session.goal_tracker.current_progress}% complete"
        
#         log_to_socket(f"{goal_status} - Made {len(session.execution_context.get('adaptations_made', []))} adaptations, {session.goal_tracker.deviation_count} course corrections")
        
#     except Exception as e:
#         logger.error(f"❌ Goal-oriented execution failed: {e}", exc_info=True)
#         log_to_socket(f"❌ Execution failed: {str(e)}")
#         await send_socket_message_to_session(session, {"type": "error", "data": str(e)})        


# async def stream_browser_view(session: UserSession):
#     """Periodically captures and sends screenshots for a specific session."""
#     while (session.current_task and not session.current_task.done() and 
#            session.browser_context and session.browser):
#         try:
#             if hasattr(session.browser_context, 'browser') and session.browser_context.browser:
#                 playwright_browser = session.browser_context.browser.playwright_browser
#                 if playwright_browser and playwright_browser.contexts:
#                     pw_context = playwright_browser.contexts[0]
#                     if pw_context and pw_context.pages:
#                         page = next((p for p in reversed(pw_context.pages) if p.url != "about:blank"), None)
#                         if page and not page.is_closed():
#                             screenshot_bytes = await page.screenshot(type="jpeg", quality=70)
#                             b64_img = base64.b64encode(screenshot_bytes).decode('utf-8')
#                             await send_socket_message_to_session(session, {"type": "stream", "data": b64_img})
#         except Exception as e:
#             logger.debug(f"Screenshot capture failed for session {session.session_id}: {e}")
        
#         await asyncio.sleep(0.5)
# In your stream_browser_view function, replace logger calls with prints:
async def stream_browser_view(session: UserSession):
    """Periodically captures and sends screenshots for a specific session with debug logging."""
    screenshot_count = 0
    consecutive_failures = 0
    
    print(f"🚀 DEBUG: Starting browser stream for session {session.session_id[:8]}", flush=True)
    
    while (session.current_task and not session.current_task.done() and 
           session.browser_context and session.browser):
        try:
            screenshot_count += 1
            print(f"📸 DEBUG: Attempting screenshot #{screenshot_count} for session {session.session_id[:8]}", flush=True)
            
            # Check browser health
            if not hasattr(session.browser_context, 'browser') or not session.browser_context.browser:
                print(f"❌ DEBUG: Browser context has no browser for session {session.session_id[:8]}", flush=True)
                break
                
            playwright_browser = session.browser_context.browser.playwright_browser
            if not playwright_browser:
                print(f"❌ DEBUG: No playwright browser for session {session.session_id[:8]}", flush=True)
                break
                
            if not playwright_browser.contexts:
                print(f"❌ DEBUG: No browser contexts for session {session.session_id[:8]}", flush=True)
                break
                
            pw_context = playwright_browser.contexts[0]
            if not pw_context or not pw_context.pages:
                print(f"⚠️ DEBUG: No pages in browser context for session {session.session_id[:8]}", flush=True)
                await asyncio.sleep(0.5)
                continue
                
            # Find active page
            page = next((p for p in reversed(pw_context.pages) if p.url != "about:blank"), None)
            if not page or page.is_closed():
                print(f"⚠️ DEBUG: No active pages or page closed for session {session.session_id[:8]}", flush=True)
                await asyncio.sleep(0.5)
                continue
                
            print(f"📄 DEBUG: Page URL: {page.url} for session {session.session_id[:8]}", flush=True)
            
            # Attempt screenshot
            screenshot_bytes = await page.screenshot(type="jpeg", quality=70)
            b64_img = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            print(f"✅ DEBUG: Screenshot #{screenshot_count} captured ({len(screenshot_bytes)} bytes) for session {session.session_id[:8]}", flush=True)
            
            # Send to WebSocket
            await send_socket_message_to_session(session, {"type": "stream", "data": b64_img})
            print(f"📤 DEBUG: Screenshot #{screenshot_count} sent to {len(session.websockets)} WebSocket(s)", flush=True)
            
            consecutive_failures = 0
            
        except Exception as e:
            consecutive_failures += 1
            print(f"❌ DEBUG: Screenshot failed #{consecutive_failures}: {e}", flush=True)
            
            if consecutive_failures >= 5:
                print(f"💀 DEBUG: Too many failures, stopping stream for session {session.session_id[:8]}", flush=True)
                break
        
        await asyncio.sleep(0.5)
    
    print(f"🏁 DEBUG: Browser stream ended for session {session.session_id[:8]} (total: {screenshot_count})", flush=True)
    
    logger.info(f"📹 Browser stream ended for session {session.session_id[:8]} (total screenshots: {screenshot_count})")
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
    
    # Convert AgentRunRequest to ExecuteStepsRequest format
    # For single task execution, create a simple step
    simple_step = Step(
        id=1,
        action="complete_task",
        description=request.task,
        params={}
    )
    
    # Create ExecuteStepsRequest
    execute_request = ExecuteStepsRequest(
        session_id=request.session_id,
        steps=[simple_step],
        start_from=0,
        llm_provider=request.llm_provider,
        llm_model_name=request.llm_model_name,
        llm_api_key=request.llm_api_key
    )
    
    # Start task execution using run_steps_logic
    session.current_task = asyncio.create_task(run_steps_logic(execute_request, session))
    
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
        llm = await _initialize_llm_helper(
            request.llm_provider, request.llm_model_name, request.llm_api_key
        )
        
        planning_prompt = f"""Break down this task into specific, actionable steps:

Task: {request.task}

Return ONLY a JSON array of steps in this exact format:
[
  {{"id": 1, "action": "go_to_url", "description": "Navigate to the website", "params": {{"url": "https://example.com"}}}},
  {{"id": 2, "action": "search", "description": "Search for the product", "params": {{"query": "product name"}}}},
  {{"id": 3, "action": "click_element", "description": "Click on specific element", "params": {{"selector": ".class-name"}}}}
]

Available actions: go_to_url, search, click_element, input_text, extract_content, scroll_down, wait, done
Each step should be specific and executable."""

        response = await llm.ainvoke([{"role": "user", "content": planning_prompt}])
        
        # Parse and clean response
        try:
            from json_repair import repair_json
        except ImportError:
            # Fallback if json_repair not installed
            def repair_json(json_str):
                return json_str
        content = response.content.replace("```json", "").replace("```", "").strip()
        content = repair_json(content)
        steps_data = json.loads(content)
        
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
    
    # Start step execution (keeps session persistent)
    session.current_task = asyncio.create_task(run_steps_logic(request, session))
    
    return {
        "status": "Steps execution started",
        "session_id": session.session_id,
        "total_steps": len(request.steps)
    }

@app.post("/api/agent/continue-steps")
async def continue_with_more_steps(request: ExecuteStepsRequest):
    """Add and execute more steps to existing persistent session"""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.current_task and not session.current_task.done():
        raise HTTPException(status_code=409, detail="Agent already running in this session")
    
    # Mark session as persistent to keep browser alive
    session.persistent = True
    
    # Continue execution with existing memory and browser
    session.current_task = asyncio.create_task(run_steps_logic(request, session))
    
    # Show which features are enabled (same as execute-steps)
    enabled_features = []
    if getattr(request, 'enable_popup_killer', True):
        enabled_features.append("popup_killer")
    if getattr(request, 'enable_captcha_avoidance', True):
        enabled_features.append("captcha_avoidance") 
    if getattr(request, 'force_duckduckgo', True):
        enabled_features.append("duckduckgo_only")
    if getattr(request, 'generate_analysis', False):
        enabled_features.append("analysis_report")
    if getattr(request, 'export_formats', []):
        enabled_features.extend([f"{fmt}_export" for fmt in request.export_formats])
    
    return {
        "status": "Continuing with additional steps (session persistent)",
        "session_id": request.session_id,
        "new_steps": len(request.steps),
        "features_enabled": enabled_features,
        "existing_memory": len(getattr(session, 'execution_context', {}).get('memory_chain', ''))
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
    """Get comprehensive session status for multi-user monitoring"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate execution progress
    execution_context = getattr(session, 'execution_context', {})
    step_results = execution_context.get('step_results', [])
    
    return {
        "session_id": session.session_id,
        "session_info": {
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "persistent": session.persistent,
            "stopped": session.stopped
        },
        "browser_status": {
            "has_browser": session.browser is not None,
            "has_context": session.browser_context is not None,
            "browser_ready": execution_context.get('browser_ready', False),
            "current_url": execution_context.get('current_url', '')
        },
        "execution_status": {
            "task_running": session.current_task is not None and not session.current_task.done(),
            "steps_completed": len(session.executed_steps),
            "current_step_index": session.current_step_index,
            "variables": execution_context.get('variables', {}),
            "last_action": execution_context.get('last_action', None)
        },
        "connection_status": {
            "websocket_count": len(session.websockets),
            "active_connections": len([ws for ws in session.websockets if not ws.client_state.DISCONNECTED])
        },
        "recent_results": step_results[-3:] if step_results else [],  # Last 3 step results
        "performance": {
            "avg_step_time": sum(
                step.get('execution_time_seconds', 0) 
                for step in session.executed_steps
            ) / max(len(session.executed_steps), 1),
            "total_execution_time": sum(
                step.get('execution_time_seconds', 0) 
                for step in session.executed_steps
            )
        }
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
    streamer_task = None
    
    logger.info(f"🔌 WebSocket connected for session {session_id[:8]}")
    
    try:
        # Send immediate connection message
        await websocket.send_text(json.dumps({
            "type": "connected", 
            "data": f"Connected to session {session_id[:8]}"
        }))
        
        # Send session status
        execution_context = getattr(session, 'execution_context', {})
        await websocket.send_text(json.dumps({
            "type": "log", 
            "data": f"Session ready - Browser: {'✅' if session.browser else '❌'}, Steps completed: {len(getattr(session, 'executed_steps', []))}"
        }))
        
        # Wait a moment for agent to potentially start
        await asyncio.sleep(1)
        
        # Start browser streaming if task is running
        if session.current_task and not session.current_task.done():
            logger.info(f"📹 Starting browser stream for session {session_id[:8]}")
            streamer_task = asyncio.create_task(stream_browser_view(session))
        
        # Keep connection alive while task is running
        while session.current_task and not session.current_task.done():
            await asyncio.sleep(1)
        
        # Send final message when task completes
        await websocket.send_text(json.dumps({
            "type": "log", 
            "data": "✅ Session execution completed"
        }))

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected for session {session_id[:8]}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for session {session_id[:8]}: {e}")
    finally:
        # Clean up streaming task
        if streamer_task and not streamer_task.done():
            streamer_task.cancel()
            logger.info(f"📹 Stopped browser stream for session {session_id[:8]}")
        
        # Remove from session
        if websocket in session.websockets:
            session.websockets.remove(websocket)
        
        logger.info(f"🔌 WebSocket cleanup complete for session {session_id[:8]}")

# --- Main Execution ---
def main():
    """Main entry point"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7788"))
    
    logger.info(f"Starting multi-user Browser Use API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()


























