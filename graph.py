import os
from functools import partial

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


def should_continue(state: AgentState) -> str:
    if state["classification"]:
        return state["classification"]
    return "end"


workflow = StateGraph(AgentState)

# --- Node Definitions ---
router_node_partial = partial(router_node, llm_router=llm_router, profile=profile)
echo_node_partial = partial(echo_node, llm=llm, profile=profile)
philosopher_node_partial = partial(philosopher_node, llm=llm, profile=profile)
reflector_node_partial = partial(reflector_node, llm=llm, profile=profile)
roleplay_node_partial = partial(roleplay_node, llm=llm, profile=profile)
planner_node_partial = partial(planner_node, llm=llm)
executor_node_partial = partial(executor_node, llm=llm)

# Add all nodes to the graph
workflow.add_node("memory_retriever", memory_retrieval_node)
workflow.add_node("router", router_node_partial)
workflow.add_node("echo_respond", echo_node_partial)
workflow.add_node("philosopher", philosopher_node_partial)
workflow.add_node("reflector", reflector_node_partial)
workflow.add_node("roleplay", roleplay_node_partial)
workflow.add_node("planner", planner_node_partial)
workflow.add_node("executor", executor_node_partial)
workflow.add_node("save_memories", save_memories_node)

# --- Edge Definitions ---
# 1. Start with memory retrieval
workflow.set_entry_point("memory_retriever")
workflow.add_edge("memory_retriever", "router")

# 2. Route to the appropriate persona node
workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "echo_respond": "echo_respond",
        "philosopher": "philosopher",
        "reflector": "reflector",
        "roleplay": "roleplay",
        "complex_reasoning": "planner",
        "end": END,
    },
)

# 3. For complex reasoning, follow the plan -> execute path
workflow.add_edge("planner", "executor")

# 4. After any response is generated, save the memories
workflow.add_edge("echo_respond", "save_memories")
workflow.add_edge("philosopher", "save_memories")
workflow.add_edge("reflector", "save_memories")
workflow.add_edge("roleplay", "save_memories")
workflow.add_edge("executor", "save_memories")

# 5. The final step is to end the graph
workflow.add_edge("save_memories", END)

# --- Compile the App ---
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
