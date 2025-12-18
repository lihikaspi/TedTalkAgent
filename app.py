import streamlit as st
from agent import initialize_agent

# --- Page Config ---
st.set_page_config(page_title="TED Talk Expert", page_icon="🔴")

# Custom Styling
st.markdown("""
    <style>
        .stApp { max-width: 800px; margin: 0 auto; }
        .stButton>button { background-color: #e62b1e; color: white; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🔴 TED Talk Assistant")
st.info("I answer questions based strictly on the TED dataset context.")


# --- Initialize Agent ---
# We cache the agent resource so it doesn't re-initialize on every streamlit rerun
@st.cache_resource
def load_agent():
    return initialize_agent()


agent_with_history = load_agent()

# --- Session State Management ---
# Streamlit clears the screen on every interaction, so we store the chat UI history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Logic ---
if user_query := st.chat_input("Ask me something about TED talks..."):
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # We call the 'invoke' method from the agent initialized in agent.py
                # Note: 'test_session' ensures the agent's internal memory persists
                response = agent_with_history.invoke(
                    {"input": user_query},
                    config={"configurable": {"session_id": "streamlit_session"}}
                )

                output_text = response["output"]
                st.markdown(output_text)

                # Add to UI history
                st.session_state.messages.append({"role": "assistant", "content": output_text})

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This assistant uses RAG (Retrieval-Augmented Generation) to search through TED talk transcripts and summaries.")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        # We don't need to manually clear the agent's memory here because
        # StreamlitChatMessageHistory is not used; the internal history
        # is tied to the session_id.
        st.rerun()