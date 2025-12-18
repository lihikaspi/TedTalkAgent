import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Add the parent directory to the path so we can import agent.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import initialize_agent

app = FastAPI()

class ChatRequest(BaseModel):
    input: str
    session_id: Optional[str] = "default_session"

# Initialize agent as a global variable
agent_with_history = initialize_agent()

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Re-using the logic from your agent.py
        response = agent_with_history.invoke(
            {"input": request.input},
            config={"configurable": {"session_id": request.session_id}}
        )
        return {"output": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))