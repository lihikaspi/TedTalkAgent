import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import initialize_agent, vectorstore
from config import CHUNK_SIZE, OVERLAP_PERCENT, TOP_K, SYSTEM_PROMPT

app = FastAPI()


# Updated Request Model to accept session_id from the frontend
class PromptRequest(BaseModel):
    question: str
    session_id: Optional[str] = "api_user_session"


# Response Models
class ContextItem(BaseModel):
    talk_id: str
    title: str
    chunk: str
    score: float


class AugmentedPrompt(BaseModel):
    System: str
    User: str


class PromptResponse(BaseModel):
    response: str
    context: List[ContextItem]
    Augmented_prompt: AugmentedPrompt


# Initialize the agent
agent_executor = initialize_agent()


@app.get("/api/stats")
def get_stats():
    """Returns the current RAG configuration."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_PERCENT,
        "top_k": TOP_K
    }


@app.post("/api/prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest):
    try:
        # 1. Retrieve context with scores
        docs_with_scores = vectorstore.similarity_search_with_score(request.question, k=TOP_K)

        context_list = []
        context_text_for_user_prompt = ""

        for doc, score in docs_with_scores:
            item = ContextItem(
                talk_id=str(doc.metadata.get("talk_id", "")),
                title=doc.metadata.get("title", "Unknown"),
                chunk=doc.page_content,
                score=float(score)
            )
            context_list.append(item)
            context_text_for_user_prompt += f"\n---\n{doc.page_content}"

        # 2. Run the agent using the session_id sent from the frontend
        # This allows the "New Chat" button to clear memory by sending a new ID
        result = agent_executor.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )

        # 3. Construct the Augmented_prompt view
        augmented_user = f"Contextual Data:{context_text_for_user_prompt}\n\nQuestion: {request.question}"

        return {
            "response": result["output"],
            "context": context_list,
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": augmented_user
            }
        }

    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)