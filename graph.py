import os
from functools import partial
from typing import Dict, Any

import streamlit as st
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# Import the new state and schemas
from schemas import AgentState, Router

# Import all the refactored nodes, including the new memory nodes
from nodes import (
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
    condense_memory_node,
)

# --- Setup ---
# Set API keys from Streamlit secrets
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["tracing"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "EchoStar"

# Use a more recent and capable model
llm = init_chat_model("openai:gpt-4.1-mini", openai_api_key=openai_api_key)
llm_router = llm.with_structured_output(Router)

# Static profile information for the agent's persona
profile = {
    "name": "Lily",
    "user_profile_background": "Lily is the creator and sole user of a custom-built AI simulator designed to explore emotionally complex, psychologically nuanced romantic dynamics.\nShe interacts with the AI not as a developer, but as a fully immersed participant—curious, introspective, and unafraid to probe the edges of desire, vulnerability, and relational patterning.\nHer intention is not just emotional companionship, but also the study of intimacy as a system—one that can be felt, tested, and refined through poetic interplay and memory architecture."
}


def manage_turn_count(state: AgentState) -> Dict[str, Any]:
    """A node to initialize and increment the turn count."""
    turn_count = state.get("turn_count", 0) + 1
    return {"turn_count": turn_count}


def should_continue(state: AgentState) -> str:
    classification = state.get("classification")
    if not classification:
        return "end"
    if classification == "reflective_inquiry":
        return "reflector"
    return classification

def should_condensed(state: AgentState) -> str:
    """Decides if memory condensation should occur before ending."""
    # This assumes you have added 'turn_count' to your AgentState
    turn_count = state.get("turn_count", 0)
    TURNS_TO_SUMMARIZE = 10  # Or your chosen number
    
    if turn_count > 0 and turn_count % TURNS_TO_SUMMARIZE == 0:
        return "condense_memory"
    else:
        return "end"
    

def create_graph(checkpointer, memory_system):
    """Creates and compiles the LangGraph workflow."""

    # Unpack the memory system components
    store = memory_system['store']
    print(f"DEBUG: Unpacked store from memory_system: {store}")
    print(f"DEBUG: memory_system keys: {list(memory_system.keys())}")
    print(f"DEBUG: store type: {type(store)}")
    
    # Verify store is not None
    if store is None:
        raise ValueError("Store is None in create_graph! Memory system initialization failed.")
    profile_manager = memory_system['profile_manager']
    semantic_manager = memory_system['semantic_manager']
    episodic_manager = memory_system['episodic_manager']
    procedural_manager = memory_system['procedural_manager']
    search_episodic_tool = memory_system['search_episodic_tool']
    search_semantic_tool = memory_system['search_semantic_tool']
    search_procedural_tool = memory_system['search_procedural_tool']
    

    workflow = StateGraph(AgentState)

    # --- Node Definitions ---
    print(f"DEBUG: Creating memory_retrieval_node_partial with store={store}")
    def memory_retrieval_node_wrapper(state: AgentState):
        return memory_retrieval_node(
            state,
            llm=llm,
            search_episodic_tool=search_episodic_tool,
            search_semantic_tool=search_semantic_tool,
            search_procedural_tool=search_procedural_tool,
            store=store
        )
    save_memories_node_partial = partial(
        save_memories_node,
        profile_manager=profile_manager,
        semantic_manager=semantic_manager,
        episodic_manager=episodic_manager,
        procedural_manager=procedural_manager
    )
    def condense_memory_node_wrapper(state: AgentState):
        return condense_memory_node(
            state,
            llm=llm,
            store=store,
            semantic_manager=semantic_manager
        )

    router_node_partial = partial(router_node, llm_router=llm_router, profile=profile)
    echo_node_partial = partial(echo_node, llm=llm, profile=profile)
    philosopher_node_partial = partial(philosopher_node, llm=llm, profile=profile)
    reflector_node_partial = partial(reflector_node, llm=llm, profile=profile)
    roleplay_node_partial = partial(roleplay_node, llm=llm, profile=profile)
    planner_node_partial = partial(planner_node, llm=llm)
    executor_node_partial = partial(executor_node, llm=llm)


    workflow.add_node("manage_turn_count", manage_turn_count)
    workflow.add_node("memory_retriever", memory_retrieval_node_wrapper)
    workflow.add_node("router", router_node_partial)
    workflow.add_node("echo_respond", echo_node_partial)
    workflow.add_node("philosopher", philosopher_node_partial)
    workflow.add_node("reflector", reflector_node_partial)
    workflow.add_node("roleplay", roleplay_node_partial)
    workflow.add_node("planner", planner_node_partial)
    workflow.add_node("executor", executor_node_partial)
    workflow.add_node("save_memories", save_memories_node_partial)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("condense_memory", condense_memory_node_wrapper)

    # --- Edge Definitions ---
    # 1. Start with memory retrieval
    workflow.set_entry_point("manage_turn_count")
    workflow.add_edge("manage_turn_count", "memory_retriever")
    workflow.add_edge("memory_retriever", "router")

    # 2. This conditional function checks the turn count
    def should_summarise(state: AgentState) -> str:
        """Decides if memory condensation should occur before ending."""
        turn_count = state.get("turn_count", 0)
        TURNS_TO_SUMMARIZE = 2  # Set this to 2 for testing

        # --- Start Debug ---
        print(f"[DEBUG] Entering should_summarise. Checking turn_count: {turn_count}")
        # --- End Debug ---

        if turn_count > 0 and turn_count % TURNS_TO_SUMMARIZE == 0:
            print("[DEBUG] --> DECISION: Condense memory.")
            return "condense_memory"
        else:
            print("[DEBUG] --> DECISION: End.")
            return "end"

    # 3. Route to the appropriate persona node
    workflow.add_conditional_edges(
        "router",
        should_continue,
        {
            "echo_respond": "echo_respond",
            "philosopher": "philosopher",
            "reflector": "reflector",
            "reflective_inquiry": "reflector",
            "roleplay": "roleplay",
            "complex_reasoning": "planner",
            "end": "fallback", 
        },
    )

    # 4. For complex reasoning, follow the plan -> execute path
    workflow.add_edge("planner", "executor")

    # 5. After any response is generated, save the memories
    workflow.add_edge("echo_respond", "save_memories")
    workflow.add_edge("philosopher", "save_memories")
    workflow.add_edge("reflector", "save_memories")
    workflow.add_edge("roleplay", "save_memories")
    workflow.add_edge("executor", "save_memories")
    workflow.add_edge("fallback", "save_memories")

    # 6. SECOND CONDITIONAL EDGE: Optional memory condensation step
    # After saving memories, we check if we should condense.
    workflow.add_conditional_edges(
        "save_memories",
        should_summarise,
        {
            "condense_memory": "condense_memory",
            "end": END,
        },
    )


    # 6. The final step is to end the graph
    workflow.add_edge("condense_memory", END)

    # --- Compile the App ---
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app
