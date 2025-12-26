import os
from config import (OPENAI_API_KEY, MODEL_BASE_URL, CHAT_MODEL, EMBED_MODEL, PC_API_KEY,
                    INDEX_NAME, SYSTEM_PROMPT, TOP_K)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Initialize components globally for reuse
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    base_url=MODEL_BASE_URL,
    model=EMBED_MODEL
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PC_API_KEY
)


def initialize_agent():
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    @tool
    def search_ted_talks(query: str) -> str:
        """Searches the TED talk database for relevant transcripts and metadata."""
        docs = retriever.invoke(query)
        formatted_docs = []
        for doc in docs:
            title = doc.metadata.get('title', 'Unknown Title')
            speaker = doc.metadata.get('speaker', 'Unknown Speaker')
            formatted_docs.append(f"Source: {title} by {speaker}\nContent: {doc.page_content}\n---")
        return "\n".join(formatted_docs)

    tools = [search_ted_talks]
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, base_url=MODEL_BASE_URL, model=CHAT_MODEL)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    history = ChatMessageHistory()

    return RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )