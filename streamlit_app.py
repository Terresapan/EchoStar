import streamlit as st
from uuid import uuid4
from utils import save_feedback, check_password
from graph import create_graph
from memory import get_memory_store
from langgraph.checkpoint.memory import MemorySaver

# Setup sidebar with instructions and feedback form
def setup_sidebar():
    """Sets up the Streamlit sidebar with app information, instructions, and technical details."""
    
    st.sidebar.header("EchoStar AI Simulator 🌘")
    st.sidebar.divider()

    st.sidebar.header("Instructions for Lily")
    st.sidebar.markdown(
        "1. **Initiate Dialogue**: Begin a conversation with AI YanGG by typing in the chat box.\n"
        "2. **Observe Personas**: AI YanGG's response style will change based on the conversation's tone (e.g., casual, philosophical, roleplay).\n"
        "3. **Provide Feedback**: You can shape AI YanGG's behavior. If you want it to act differently, provide direct feedback (e.g., *'From now on, only respond with questions'*). This creates a procedural memory."
    )
    
    st.sidebar.divider()

    st.sidebar.header("How AI YanGG's Memory Works")
    st.sidebar.markdown(
        "AI YanGG's memory is powered by a `LangGraph` engine and is more than a simple chat history. It's composed of several interconnected systems:"
    )
    
    # --- UPDATED SECTION ---
    st.sidebar.subheader("🧠 Hierarchical Memory (Episodic & Semantic)")
    st.sidebar.markdown(
        "AI YanGG remembers the *what* and the *meaning* of your past conversations. To maintain efficiency, it periodically **summarizes and consolidates** older dialogues into higher-level insights, creating a true memory hierarchy instead of a flat timeline."
    )
    
    st.sidebar.subheader("⚙️ Procedural Memory")
    st.sidebar.markdown(
        "AI YanGG learns *how to act*. When you provide direct feedback, it creates a behavioral rule that modifies its core personality, allowing it to adapt its communication style to your preferences."
    )

    # --- NEW SECTION ---
    st.sidebar.subheader("🔗 Connective Memory (Reflection Mode)")
    st.sidebar.markdown(
        "When you share something emotionally vulnerable or philosophically deep, the agent can enter a special 'Reflection Mode'. It will proactively search its entire memory history to find connections between your present feelings and past events, offering deeper, more insightful responses."
    )
    
    st.sidebar.subheader("📝 Short-Term Working Memory")
    st.sidebar.markdown(
        "For complex, analytical questions, AI YanGG uses a 'scratchpad' to outline a multi-step plan before answering, enabling more thoughtful and structured reasoning."
    )


    st.sidebar.write("### 🎧 Listen to our Podcast for more insights")
    st.sidebar.markdown(
        "[AI YanGG Simulator Podcast Link](https://open.spotify.com/episode/4ZMxA2xlKMbIOxcdb3SJEv)"
    )


    st.sidebar.write("### 🌎 Visit my AI Agent Projects Website")
    st.sidebar.markdown(
        "[Terresa Pan's Agent Garden Link](https://ai-agents-garden.lovable.app/)"
    )

    # Feedback section
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""

    st.sidebar.markdown("---")
    st.sidebar.subheader("💭 Feedback")
    feedback = st.sidebar.text_area(
        "Share your thoughts",
        value=st.session_state.feedback,
        placeholder="Your feedback helps us improve..."
    )

    if st.sidebar.button("📤 Submit Feedback"):
        if feedback:
            try:
                save_feedback(feedback)
                st.session_state.feedback = ""
                st.sidebar.success("✨ Thank you for your feedback!")
            except Exception as e:
                st.sidebar.error(f"❌ Error saving feedback: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please enter feedback before submitting")

    try:
        st.sidebar.image("assets/bot01.jpg", use_container_width=True)
    except:
        pass

def main():
    """Main application function."""
    # Set page configuration (must be the first Streamlit command)
    st.set_page_config(
        page_title="EchoStar AI Simulator 🌘",
        page_icon="🌘",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    setup_sidebar()

    if not check_password():
        st.stop()

    # --- STATE INITIALIZATION ---
    # This block runs only ONCE per session
    if "memory_system" not in st.session_state:
        try:
            st.session_state.memory_system = get_memory_store()
        except Exception as e:
            st.error(f"❌ Failed to initialize memory system: {str(e)}")
            st.error("Please check your OpenAI API key and network connection.")
            st.stop()
    
    if "app" not in st.session_state:
        try:
            # Use a single, persistent checkpointer
            checkpointer = MemorySaver()
            st.session_state.app = create_graph(checkpointer=checkpointer, memory_system=st.session_state.memory_system)
        except Exception as e:
            st.error(f"❌ Failed to initialize application graph: {str(e)}")
            st.stop()

    # Use the persistent objects from session_state from now on
    app = st.session_state.app
    store = st.session_state.memory_system["store"]
    # ---------------------------

    st.markdown("<h1 class='main-title'>EchoStar AI Simulator 🌘</h1>", unsafe_allow_html=True)
    st.markdown("""
    This is a custom-built AI simulator for its user, Lily, designed to explore emotionally complex and psychologically nuanced dynamics with the AI YanGG.
    """)

    # Initialize session state for messages and thread_id
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    prompt = st.chat_input("💭 Say something to YanGG")
    if prompt:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Define the config for this invocation
        config = {"configurable": {"user_id": "Lily", "thread_id": st.session_state.thread_id}}

        # Invoke the LangGraph workflow with just the new message
        result = app.invoke({"message": prompt}, config=config) # type: ignore

        # Display the agent's response
        with st.chat_message("assistant"):
            st.write(result["response"])
        # Add agent message to session state
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})

        # Display memory from the persistent store object
        try:
            st.subheader("Agent Memory (Profile)")
            st.write(store.search(('echo_star', 'Lily', 'profile')))

            st.subheader("Agent Memory (Semantic)")
            st.write(store.search(('echo_star', 'Lily', 'facts')))
            
            st.subheader("Agent Memory (Episodic)")
            st.write(store.search(('echo_star', 'Lily', 'collection')))

            st.subheader("Agent Memory (Procedural)")
            st.write(store.search(('echo_star', 'Lily', 'rules')))
        except Exception as e:
            st.error(f"❌ Error displaying memory: {str(e)}")
            st.info("Memory display failed, but the conversation should continue normally.")


if __name__ == "__main__":
    main()
