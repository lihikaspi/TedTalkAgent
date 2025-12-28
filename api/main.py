import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Vercel Path Configuration ---
# Ensures the root directory is in sys.path so local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now we can safely import from the root directory
from agent import initialize_agent, embeddings
from config import CHUNK_SIZE, OVERLAP_PERCENT, TOP_K, SYSTEM_PROMPT, INDEX_NAME, PC_API_KEY
from langchain_pinecone import PineconeVectorStore

app = FastAPI()

_agent_executor = None
_vectorstore = None


def get_rag_resources():
    """Initializes and returns the agent and vector store if not already loaded."""
    global _agent_executor, _vectorstore

    if _agent_executor is None:
        _agent_executor = initialize_agent()

    if _vectorstore is None:
        # We re-establish the connection to the vector store for context retrieval
        _vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=PC_API_KEY
        )
    return _agent_executor, _vectorstore


# --- Data Models ---
class PromptRequest(BaseModel):
    question: str
    session_id: Optional[str] = "api_user_session"


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


# --- Endpoints ---

@app.get("/api/stats")
def get_stats():
    """Returns the current RAG configuration for debugging."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_PERCENT,
        "top_k": TOP_K
    }


@app.post("/api/prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest):
    try:
        # 1. Ensure resources are initialized
        executor, v_store = get_rag_resources()

        # 2. Retrieve context with scores directly from vectorstore
        docs_with_scores = v_store.similarity_search_with_score(request.question, k=TOP_K)

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

        # 3. Run the agent logic with history management
        result = executor.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )

        # 4. Construct the view of the augmented prompt for the frontend
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
        # Log to Vercel console
        print(f"Deployment API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")


# Local testing entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)