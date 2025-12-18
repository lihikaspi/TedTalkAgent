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


def initialize_agent():
    """
    Initializes the TED RAG Agent designed to handle specific retrieval goals:
    1. Precise Fact Retrieval
    2. Multi-Result Topic Listing (Up to 3)
    3. Key Idea Summary Extraction
    4. Recommendation with Justification
    """
    # Setup Embeddings
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=MODEL_BASE_URL,
        model=EMBED_MODEL
    )

    # Setup Pinecone connection
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PC_API_KEY
    )

    # Use the vectorstore as a retriever with TOP_K from config
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    @tool
    def search_ted_talks(query: str) -> str:
        """
        Searches the TED talk database for relevant transcripts, summaries, and metadata.
        Input should be a search query string based on the user's intent.
        Use this to find specific facts, list multiple talks, or get details for summaries.
        """
        docs = retriever.invoke(query)

        formatted_results = []
        for doc in docs:
            title = doc.metadata.get('title', 'Unknown Title')
            speaker = doc.metadata.get('speaker', 'Unknown Speaker')
            topics = doc.metadata.get('topics', 'N/A')
            url = doc.metadata.get('url', 'N/A')
            content = doc.page_content

            result_str = (
                f"TALK TITLE: {title}\n"
                f"SPEAKER: {speaker}\n"
                f"URL: {url}\n"
                f"TOPICS: {topics}\n"
                f"CONTENT/SUMMARY SNIPPET: {content}\n"
                f"---"
            )
            formatted_results.append(result_str)

        if not formatted_results:
            return "No relevant TED talks found in the dataset for this query."

        return "\n".join(formatted_results)

    tools = [search_ted_talks]

    # Initialize the Chat Model
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=MODEL_BASE_URL,
        model=CHAT_MODEL
    )

    # Build the Prompt Template using the SYSTEM_PROMPT from config
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Construct the tool-calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Setup the Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    # Setup persistent chat history for the session
    history = ChatMessageHistory()

    return RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def run_chat():
    """Starts the interactive terminal session for the TED RAG Agent."""
    agent = initialize_agent()
    print("\n==========================================")
    print("   TED DATASET ASSISTANT INITIALIZED")
    print(f"   (Retrieval Depth: {TOP_K} chunks)")
    print("==========================================\n")

    session_id = "default_user_session"

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "q"]:
            break

        if not user_query.strip():
            continue

        try:
            response = agent.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\nAssistant: {response['output']}\n")
        except Exception as e:
            print(f"\n[Error Encountered]: {str(e)}\n")


if __name__ == "__main__":
    run_chat()