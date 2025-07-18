import streamlit as st
from uuid import uuid4
from src.utils.utils import save_feedback, check_password
from src.agents.graph import create_graph
from src.agents.memory import create_memory_system
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Import configuration manager
from config.manager import get_config_manager

# Import logging and validation utilities
from src.utils.logging_utils import get_logger, validate_user_input, setup_application_logging

# Initialize logger for this module
logger = get_logger(__name__)

# Setup application logging
setup_application_logging()

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
        "[AI YanGG Simulator Podcast Link](https://open.spotify.com/episode/27Cepxa1bI3tTN9G3fhJbk)"
    )


    st.sidebar.write("### 🌎 Visit my AI Agent Projects Website")
    st.sidebar.markdown(
        "[Terresa Pan's Agent Garden Link](https://agentgarden.lovable.app/)"
    )

    
    st.sidebar.image("assets/bot01.jpg", use_container_width=True)
    

def main():
    """Main application function."""
    # Set page configuration (must be the first Streamlit command)
    st.set_page_config(
        page_title="EchoStar AI Simulator 🌘",
        page_icon="🌘",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if not check_password():
        st.stop()

    setup_sidebar()

    # --- CONFIGURATION VALIDATION ---
    # Validate configuration at startup
    if "config_validated" not in st.session_state:
        try:
            config_manager = get_config_manager()
            if not config_manager.validate_config():
                st.error("❌ Configuration validation failed!")
                st.error("Please check your configuration files and environment variables.")
                st.stop()
            
            # Display configuration info in development
            config = config_manager.config
            if config and config.logging.level == "DEBUG":
                st.info("✅ Configuration loaded successfully")
                with st.expander("Configuration Details"):
                    st.json({
                        "memory": {
                            "turns_to_summarize": config.memory.turns_to_summarize,
                            "search_limit": config.memory.search_limit,
                            "enable_condensation": config.memory.enable_condensation
                        },
                        "routing": {
                            "roleplay_threshold": config.routing.roleplay_threshold,
                            "enable_fallback": config.routing.enable_fallback
                        },
                        "llm": {
                            "model_name": config.llm.model_name,
                            "temperature": config.llm.temperature,
                            "timeout": config.llm.timeout
                        },
                        "logging": {
                            "level": config.logging.level,
                            "format": config.logging.format
                        }
                    })
            
            st.session_state.config_validated = True
            
        except Exception as e:
            st.error(f"❌ Configuration initialization failed: {str(e)}")
            st.error("Please check your configuration files and try again.")
            st.stop()

    # --- STATE INITIALIZATION ---
    # This block runs only ONCE per session
    if "memory_system" not in st.session_state:
        try:
            # Initialize the memory store and system
            store = InMemoryStore()
            st.session_state.memory_system = create_memory_system(store)
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
        # Validate user input before processing
        logger.info("User message received", 
                   message_length=len(prompt),
                   thread_id=st.session_state.thread_id)
        
        validation_result = validate_user_input(prompt)
        
        if not validation_result.is_valid:
            # Display validation errors to user
            logger.warning("User input validation failed", 
                          errors=validation_result.error_messages,
                          warnings=validation_result.warnings)
            
            st.error("❌ Message validation failed:")
            for error in validation_result.errors:
                st.error(f"• {error.message}")
            
            if validation_result.warnings:
                st.warning("⚠️ Warnings:")
                for warning in validation_result.warnings:
                    st.warning(f"• {warning}")
            
            return  # Don't process invalid input
        
        # Log any warnings but continue processing
        if validation_result.has_warnings:
            logger.info("User input has warnings but is valid", 
                       warnings=validation_result.warnings)
            for warning in validation_result.warnings:
                st.info(f"ℹ️ {warning}")
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Define the config for this invocation
        config = {"configurable": {"user_id": "Lily", "thread_id": st.session_state.thread_id}}

        try:
            # Invoke the LangGraph workflow with just the new message
            logger.info("Starting LangGraph workflow execution", 
                       thread_id=st.session_state.thread_id)
            
            with logger.performance_timer("langgraph_execution", thread_id=st.session_state.thread_id):
                result = app.invoke({"message": prompt}, config=config) # type: ignore
            
            logger.info("LangGraph workflow completed successfully", 
                       response_length=len(result.get("response", "")),
                       thread_id=st.session_state.thread_id)

            # Display the agent's response
            with st.chat_message("assistant"):
                st.write(result["response"])
            # Add agent message to session state
            st.session_state.messages.append({"role": "assistant", "content": result["response"]})
            
        except Exception as e:
            logger.error("LangGraph workflow execution failed", 
                        error=e,
                        thread_id=st.session_state.thread_id)
            st.error(f"❌ Error processing your message: {str(e)}")
            st.info("Please try again or rephrase your message.")

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
