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
from pinecone import Pinecone

# Initialize embeddings globally
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    base_url=MODEL_BASE_URL,
    model=EMBED_MODEL
)


def initialize_agent():
    # Verify index existence before connecting to avoid 404 crashes
    pc = Pinecone(api_key=PC_API_KEY)
    active_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in active_indexes:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' not found. Please run 'prepare_embeds.py' "
            "successfully before starting the agent."
        )

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PC_API_KEY
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    @tool
    def search_ted_talks(query: str) -> str:
        """
        Searches the TED talk database.
        Returns high-level summaries and specific transcript snippets.
        Essential for finding facts, listing titles, and providing evidence.
        """
        docs = retriever.invoke(query)

        formatted_results = []
        for doc in docs:
            title = doc.metadata.get('title', 'Unknown')
            speaker = doc.metadata.get('speaker', 'Unknown')
            c_type = doc.metadata.get('content_type', 'content')

            # Labeling type helps with Project Goals 2 (listing) and 3 (summary)
            prefix = f"[{c_type.upper()}] "
            entry = f"Source: {title} by {speaker}\n{prefix}Content: {doc.page_content}\n---"
            formatted_results.append(entry)

        if not formatted_results:
            return "No relevant TED talks found in the retrieved context."

        return "\n".join(formatted_results)

    tools = [search_ted_talks]

    # temperature removed for GPT-5 compatibility; using provider defaults
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=MODEL_BASE_URL,
        model=CHAT_MODEL
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    history = ChatMessageHistory()

    return RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )