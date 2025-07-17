import os
from functools import partial
from typing import Dict, Any, Optional

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# Import the new state and schemas
from .schemas import AgentState, Router

# Import configuration manager
from config.manager import get_config_manager

# Import logging utilities
from ..utils.logging_utils import get_logger

# Import nodes
from .nodes import (
    manage_turn_count,
    memory_retrieval_node,
    router_node,
    echo_node,
    philosopher_node,
    reflector_node,
    roleplay_node,
    planner_node,
    executor_node,
    save_memories_node,
    fallback_node,
    condense_memory_node
)

logger = get_logger(__name__)

def _setup_environment():
    """Setup environment variables for LangChain tracing."""
    try:
        # Only set up Streamlit secrets if running in Streamlit context
        if hasattr(st, 'secrets') and 'general' in st.secrets:
            openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = st.secrets["tracing"]["LANGCHAIN_API_KEY"]
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_PROJECT"] = "EchoStar"
    except Exception:
        # If not in Streamlit context, environment should already be set up
        pass

def _initialize_llm():
    """Initialize the LLM with configuration."""
    # Get OpenAI API key - try Streamlit secrets first, then environment variable
    openai_api_key = None
    
    try:
        if hasattr(st, 'secrets') and 'general' in st.secrets and 'OPENAI_API_KEY' in st.secrets['general']:
            openai_api_key = st.secrets['general']['OPENAI_API_KEY']
    except Exception:
        pass
    
    # Get configuration manager
    config_manager = get_config_manager()

    # Use configured model
    llm_config = config_manager.get_llm_config()
    llm = init_chat_model(
        llm_config.model_name, 
        openai_api_key=openai_api_key,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        timeout=llm_config.timeout
    )
    llm_router = llm.with_structured_output(Router)
    
    return llm, llm_router


def should_condense_memory(state: AgentState) -> str:
    """Determine if memory should be condensed based on turn count."""
    turn_count = state.get("turn_count", 0)
    config_manager = get_config_manager()
    memory_config = config_manager.get_memory_config()
    TURNS_TO_SUMMARIZE = memory_config.turns_to_summarize
    
    logger.info("Checking memory condensation condition", 
                turn_count=turn_count, 
                threshold=TURNS_TO_SUMMARIZE)
    
    if turn_count and turn_count % TURNS_TO_SUMMARIZE == 0:
        logger.info("Memory condensation triggered", turn_count=turn_count)
        return "condense_memory"
    else:
        return "save_memories"


def route_classification(state: AgentState) -> str:
    """Route based on the classification from the router."""
    classification = state.get("classification")
    
    logger.info("Routing message", classification=classification)
    
    if classification == "echo_respond":
        return "echo_respond"
    elif classification == "roleplay":
        return "roleplay"
    elif classification == "reflector":
        return "reflector"
    elif classification == "philosopher":
        return "philosopher"
    elif classification == "complex_reasoning":
        return "complex_reasoning"
    else:
        logger.warning("Unknown classification, routing to fallback", 
                      classification=classification)
        return "fallback"


def create_graph(checkpointer: MemorySaver, memory_system: Dict[str, Any]):
    """Creates and compiles the LangGraph workflow."""
    
    logger.info("Creating agent graph", 
                memory_system_components=list(memory_system.keys()))
    
    # Setup environment variables
    _setup_environment()
    
    # Initialize LLM
    llm, llm_router = _initialize_llm()
    
    # Create the StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("manage_turn_count", manage_turn_count)
    workflow.add_node("memory_retrieval", partial(
        memory_retrieval_node,
        llm=llm,
        search_episodic_tool=memory_system["search_episodic_tool"],
        search_semantic_tool=memory_system["search_semantic_tool"],
        search_procedural_tool=memory_system["search_procedural_tool"],
        store=memory_system["store"]
    ))
    workflow.add_node("router", partial(router_node, llm_router=llm_router, profile=memory_system["profile"]))
    workflow.add_node("echo_respond", partial(echo_node, llm=llm, profile=memory_system["profile"]))
    workflow.add_node("philosopher", partial(philosopher_node, llm=llm, profile=memory_system["profile"]))
    workflow.add_node("reflector", partial(reflector_node, llm=llm, profile=memory_system["profile"]))

    workflow.add_node("roleplay", partial(roleplay_node, llm=llm, profile=memory_system["profile"]))
    workflow.add_node("complex_reasoning", partial(planner_node, llm=llm))
    workflow.add_node("planner", partial(planner_node, llm=llm))
    workflow.add_node("executor", partial(executor_node, llm=llm))
    workflow.add_node("save_memories", partial(
        save_memories_node,
        profile_manager=memory_system["profile_manager"],
        semantic_manager=memory_system["semantic_manager"],
        episodic_manager=memory_system["episodic_manager"],
        procedural_manager=memory_system["procedural_manager"],
        store=memory_system["store"]
    ))
    workflow.add_node("condense_memory", partial(
        condense_memory_node,
        llm=llm,
        store=memory_system["store"],
        semantic_manager=memory_system["semantic_manager"],
        episodic_manager=memory_system["episodic_manager"]
    ))
    workflow.add_node("fallback", fallback_node)
    
    # Set entry point
    workflow.set_entry_point("manage_turn_count")
    
    # Add edges
    workflow.add_edge("manage_turn_count", "memory_retrieval")
    workflow.add_edge("memory_retrieval", "router")
    
    # Add conditional routing from router
    workflow.add_conditional_edges(
        "router",
        route_classification,
        {
            "echo_respond": "echo_respond",
            "roleplay": "roleplay", 
            "reflector": "reflector",
            "philosopher": "philosopher",
            "complex_reasoning": "planner",
            "fallback": "fallback"
        }
    )
    
    # Connect response nodes to memory saving
    workflow.add_conditional_edges(
        "echo_respond",
        should_condense_memory,
        {
            "save_memories": "save_memories",
            "condense_memory": "condense_memory"
        }
    )
    workflow.add_conditional_edges(
        "philosopher",
        should_condense_memory,
        {
            "save_memories": "save_memories", 
            "condense_memory": "condense_memory"
        }
    )
    workflow.add_conditional_edges(
        "reflector",
        should_condense_memory,
        {
            "save_memories": "save_memories",
            "condense_memory": "condense_memory"
        }
    )
    workflow.add_conditional_edges(
        "roleplay",
        should_condense_memory,
        {
            "save_memories": "save_memories",
            "condense_memory": "condense_memory"
        }
    )
    workflow.add_conditional_edges(
        "fallback",
        should_condense_memory,
        {
            "save_memories": "save_memories",
            "condense_memory": "condense_memory"
        }
    )

    
    # Complex reasoning flow
    workflow.add_edge("complex_reasoning", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "executor",
        should_condense_memory,
        {
            "save_memories": "save_memories",
            "condense_memory": "condense_memory"
        }
    )
    
    # Memory operations to END
    workflow.add_edge("save_memories", END)
    workflow.add_edge("condense_memory", "save_memories")
    
    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Agent graph compiled successfully")
    
    return app