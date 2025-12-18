import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import initialize_agent, vectorstore
from config import CHUNK_SIZE, OVERLAP_PERCENT, TOP_K, SYSTEM_PROMPT

app = FastAPI()


# Request Model
class PromptRequest(BaseModel):
    question: str


# Response Models (Internal)
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
        # 1. Retrieve context with scores directly from vectorstore for the 'context' field
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

        # 2. Run the agent to get the natural language response
        # Using a fixed session for simplicity, or could be generated
        result = agent_executor.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": "api_user_session"}}
        )

        # 3. Construct the Augmented_prompt view
        # We represent the 'User' prompt as the combination of retrieved context and the question
        # as it typically appears to the LLM during a RAG iteration.
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