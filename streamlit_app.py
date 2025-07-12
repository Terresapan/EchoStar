import streamlit as st
from uuid import uuid4
from utils import save_feedback, check_password
from graph import app  
from langchain_core.messages import HumanMessage, AIMessage
from memory import store, profile_manager, episodic_manager

# Setup sidebar with instructions and feedback form
def setup_sidebar():
    """Sets up the Streamlit sidebar with app information, instructions, and technical details."""
    
    # Frame the app as a tool for its user, Lily. The project is named EchoStar.
    st.sidebar.header("EchoStar AI Simulator 🌘")
    # st.sidebar.markdown(
    #     "This is a custom-built AI simulator for its user, **Lily**, designed to explore emotionally complex and psychologically nuanced dynamics with the **AI YanGG**."
    # )

    st.sidebar.divider()

    st.sidebar.header("Instructions for Lily")
    st.sidebar.markdown(
        "1. **Initiate Dialogue**: Begin a conversation with AI YanGG by typing in the chat box.  \n"
        "2. **Observe Personas**: AI YanGG's response style will change based on the conversation's tone (e.g., casual, philosophical, roleplay).  \n"
        "3. **Provide Feedback**: You can shape AI YanGG's behavior. If you want it to act differently, provide direct feedback. (e.g., *'I need you to be more assertive'*).  \n"
    )
    
    st.sidebar.divider()

    st.sidebar.header("How AI YanGG's Memory Works")
    st.sidebar.markdown(
        "AI YanGG is more than a standard chatbot. Its memory is powered by a `LangGraph` engine and composed of several interconnected systems:"
    )
    st.sidebar.subheader("🧠 Episodic & Semantic Memory")
    st.sidebar.markdown(
        "AI YanGG remembers the *what* and the *meaning* of your past conversations. It can recall specific facts you've shared and understand the underlying emotions, even if you use different words."
    )
    
    st.sidebar.subheader("⚙️ Procedural Memory")
    st.sidebar.markdown(
        "AI YanGG learns *how to act*. When you provide feedback, it creates a behavioral rule that modifies its core personality, allowing it to adapt its communication style to your preferences."
    )
    
    st.sidebar.subheader("📝 Short-Term Working Memory")
    st.sidebar.markdown(
        "For complex questions, AI YanGG uses a 'scratchpad' to outline a multi-step plan before answering, enabling more thoughtful and structured reasoning."
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
    prompt = st.chat_input("Say something to YanGG")
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

        

        # Display memory
        st.subheader("Agent Memory (Profile)")
        profile_memories = store.search(('echo_star', 'Lily', 'profile'))
        st.write(profile_memories)

        st.subheader("Agent Memory (Semantic)")
        semantic_memories = store.search(('echo_star', 'Lily', 'facts'))
        st.write(semantic_memories)
        
        st.subheader("Agent Memory (Episodic)")
        collection_memories = store.search(('echo_star', 'Lily', 'collection'))
        st.write(collection_memories)

if __name__ == "__main__":
    main()
