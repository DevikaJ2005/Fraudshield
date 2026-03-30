"""
FraudShield FastAPI Server
OpenEnv-compatible REST API for fraud detection environment
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any
import json

from models import FraudCheckAction, DecisionEnum
from fraudshield_env import FraudShieldEnvironment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FraudShield",
    description="E-Commerce Fraud Detection OpenEnv",
    version="0.1.0"
)

# Global environment instance (in production, use session-based)
env = FraudShieldEnvironment(seed=42)
current_episode = None

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "fraudshield"}


# ============================================================================
# ENVIRONMENT ENDPOINTS
# ============================================================================

@app.post("/reset")
async def reset(task: str = "easy") -> Dict[str, Any]:
    """
    Reset environment for new episode
    
    Args:
        task: "easy", "medium", or "hard"
    
    Returns:
        Initial observation and metadata
    """
    global current_episode, env
    
    try:
        result = env.reset(task)
        current_episode = env.episode_id
        
        return {
            "observation": result.observation.model_dump(),
            "info": result.info,
            "episode_id": env.episode_id
        }
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a step in the environment
    
    Args:
        action_data: {
            "transaction_id": str,
            "decision": "fraud" | "legitimate",
            "confidence": float,
            "reasoning": str
        }
    
    Returns:
        Observation, reward, done flag, and info
    """
    global env
    
    try:
        # Parse action
        action = FraudCheckAction(
            transaction_id=action_data["transaction_id"],
            decision=DecisionEnum(action_data["decision"]),
            confidence=float(action_data["confidence"]),
            reasoning=str(action_data["reasoning"])
        )
        
        # Step environment
        result = env.step(action)
        
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "info": result.info
        }
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")
    except Exception as e:
        logger.error(f"Step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """
    Get current episode state
    
    Returns:
        Episode metadata
    """
    global env
    
    try:
        state = env.state()
        return state.model_dump()
    except Exception as e:
        logger.error(f"State error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# METADATA ENDPOINTS
# ============================================================================

@app.get("/info")
async def get_info() -> Dict[str, Any]:
    """Get environment information"""
    return {
        "name": "fraudshield",
        "version": "0.1.0",
        "description": "E-commerce fraud detection environment",
        "tasks": ["easy", "medium", "hard"],
        "max_steps": env.max_steps
    }


@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    """Get available tasks and their descriptions"""
    return {
        "easy": {
            "difficulty": "easy",
            "num_transactions": 60,
            "description": "Clear fraud signals - new sellers, high amounts, risky countries"
        },
        "medium": {
            "difficulty": "medium",
            "num_transactions": 200,
            "description": "Mixed signals - subtle patterns, false positives, ROC-AUC focus"
        },
        "hard": {
            "difficulty": "hard",
            "num_transactions": 350,
            "description": "Ring fraud - coordinated attacks, temporal patterns, network effects"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "service": "FraudShield OpenEnv",
        "version": "0.1.0",
        "description": "E-commerce fraud detection environment for AI agents",
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset?task=easy|medium|hard",
            "step": "POST /step",
            "state": "GET /state",
            "info": "GET /info",
            "tasks": "GET /tasks"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
