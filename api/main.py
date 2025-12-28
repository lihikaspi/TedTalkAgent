import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Vercel Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from agent import initialize_agent, embeddings
from config import CHUNK_SIZE, OVERLAP_PERCENT, TOP_K, SYSTEM_PROMPT, INDEX_NAME, PC_API_KEY
from langchain_pinecone import PineconeVectorStore

app = FastAPI()

# --- Lazy Initialization ---
_agent_executor = None
_vectorstore = None


def get_rag_resources():
    global _agent_executor, _vectorstore
    if _agent_executor is None:
        _agent_executor = initialize_agent()
    if _vectorstore is None:
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


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the index.html file from the root directory."""
    # Look for index.html in the parent directory (root)
    index_path = os.path.join(parent_dir, "index.html")
    if not os.path.exists(index_path):
        # Fallback if it's in the same folder during some local tests
        index_path = os.path.join(current_dir, "index.html")

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<html><body><h1>Error</h1><p>index.html not found at {index_path}</p></body></html>"


@app.get("/api/stats")
def get_stats():
    return {"chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_PERCENT, "top_k": TOP_K}


@app.post("/api/prompt", response_model=PromptResponse)
async def process_prompt(request: PromptRequest):
    try:
        executor, v_store = get_rag_resources()
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

        result = executor.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )

        return {
            "response": result["output"],
            "context": context_list,
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": f"Contextual Data:{context_text_for_user_prompt}\n\nQuestion: {request.question}"
            }
        }
    except Exception as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)