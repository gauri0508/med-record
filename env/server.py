"""
MedRecordAudit — FastAPI Server

Wraps the environment in HTTP endpoints:
  POST /reset  → start new episode
  POST /step   → take an action
  GET  /state  → get current state
  GET  /health → health check (required for HF Spaces)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from env.environment import MedRecordAuditEnv

app = FastAPI(
    title="MedRecordAudit",
    description="RL Environment for Medical Record Auditing — find missed diagnoses, drug interactions, and contradictions in patient histories.",
    version="0.1.0",
)

# Single environment instance
env = MedRecordAuditEnv()


# --- Request/Response Models ---

class ResetRequest(BaseModel):
    difficulty: str = Field(default="easy", description="Difficulty level: easy, medium, or hard")
    case_id: Optional[str] = Field(default=None, description="Specific case ID (e.g., 'easy_001') or null for random")


class ActionRequest(BaseModel):
    action: str = Field(description="Action type: read_record, cross_reference, flag_issue, submit_report")
    record_id: Optional[int] = Field(default=None, description="Record ID for read_record action")
    query: Optional[str] = Field(default=None, description="Search query for cross_reference action")
    type: Optional[str] = Field(default=None, description="Issue type for flag_issue action")
    description: Optional[str] = Field(default=None, description="Issue description for flag_issue action")
    evidence: Optional[list] = Field(default=None, description="List of evidence record IDs for flag_issue action")


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "MedRecordAudit"
    version: str = "0.1.0"


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint. Returns 200 if server is running."""
    return HealthResponse()


@app.post("/reset")
async def reset(request: ResetRequest):
    """
    Start a new episode.

    - **difficulty**: easy (20 records, 1 issue), medium (80 records, 3 issues), hard (150 records, 5-6 issues)
    - **case_id**: optional specific case, or random if omitted

    Returns the initial state with patient info and record index (summaries only).
    """
    try:
        state = env.reset(
            difficulty=request.difficulty,
            case_id=request.case_id,
        )
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/step")
async def step(request: ActionRequest):
    """
    Execute one action in the environment.

    Actions:
    - **read_record**: Read a specific medical record. Requires `record_id`.
    - **cross_reference**: Search medical databases. Requires `query`.
    - **flag_issue**: Flag a found issue. Requires `type`, `description`, `evidence`.
    - **submit_report**: End the episode and get final score.

    Returns: state, reward, done, info
    """
    action = {"action": request.action}

    if request.action == "read_record":
        if request.record_id is None:
            raise HTTPException(status_code=400, detail="record_id is required for read_record action")
        action["record_id"] = request.record_id

    elif request.action == "cross_reference":
        if not request.query:
            raise HTTPException(status_code=400, detail="query is required for cross_reference action")
        action["query"] = request.query

    elif request.action == "flag_issue":
        if not request.type:
            raise HTTPException(status_code=400, detail="type is required for flag_issue action")
        if not request.description:
            raise HTTPException(status_code=400, detail="description is required for flag_issue action")
        action["type"] = request.type
        action["description"] = request.description
        action["evidence"] = request.evidence or []

    elif request.action == "submit_report":
        pass

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {request.action}. Valid: read_record, cross_reference, flag_issue, submit_report"
        )

    result = env.step(action)
    return result


@app.get("/state")
async def state():
    """
    Get the current environment state.

    Returns patient info, record index, reviewed records, flagged findings,
    budget remaining, and available actions.
    """
    current_state = env.state()
    if "error" in current_state:
        raise HTTPException(status_code=400, detail=current_state["error"])
    return current_state
