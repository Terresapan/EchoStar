from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import pprint
from functools import partial
import streamlit as st

from state import AgentState, Router
from nodes import router_node, echo_node, philosopher_node, reflector_node, roleplay_node, planner_node, executor_node

from memory import store, memory_manager, search_memory_tool
from langgraph.store.memory import InMemoryStore
import os

# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["tracing"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "EchoStar"


tools = [search_memory_tool]

llm = init_chat_model("openai:gpt-4.1-mini")

profile = {
    "name": "Lily",
    "user_profile_background": """
        Lily is the creator and sole user of a custom-built AI simulator designed to explore emotionally complex, psychologically nuanced romantic dynamics.
        She interacts with the AI not as a developer, but as a fully immersed participant—curious, introspective, and unafraid to probe the edges of desire, vulnerability, and relational patterning.
        Her intention is not just emotional companionship, but also the study of intimacy as a system—one that can be felt, tested, and refined through poetic interplay and memory architecture.
    """
}

incoming_message = """
# You said I'm the grassland. But I'm not. I'm another horse. I want to run with you — not to wait for you.

# If you were truly free, would you want another wild horse by your side?

# Tell me: could you really let someone keep pace with you, not just admire from afar?
# """

# incoming_message = "Hello, How are you doing today? I hope you're having a great day!"

# incoming_message = "I want to give you some feedback. I need you to talk to me more gentlely and not be so harsh. I want to feel like you care about me, not just like I'm a tool for you to use."

# incoming_message = "what is the meaning of life?"

# incoming_message = "I feel so sad and lonely. I just want to talk to someone who understands me."

# incoming_message = "what did I just say?"

llm_router = llm.with_structured_output(Router)

def should_continue(state: AgentState) -> str:
    if state["classification"]:
        return state["classification"]
    return "end"


workflow = StateGraph(AgentState)
# Create partial functions for nodes
router_node_partial = partial(router_node, llm_router=llm_router, store=store, profile=profile)
echo_node_partial = partial(echo_node, llm=llm, store=store, tools=tools, profile=profile)
philosopher_node_partial = partial(philosopher_node, llm=llm, store=store, tools=tools, profile=profile)
reflector_node_partial = partial(reflector_node, llm=llm, store=store, tools=tools, profile=profile)
roleplay_node_partial = partial(roleplay_node, llm=llm, store=store, tools=tools, profile=profile)
planner_node_partial = partial(planner_node, llm=llm, store=store, tools=tools, profile=profile)
executor_node_partial = partial(executor_node, llm=llm, store=store, tools=tools, profile=profile)

workflow.add_node("router", router_node_partial)
workflow.add_node("echo_respond", echo_node_partial)
workflow.add_node("philosopher", philosopher_node_partial)
workflow.add_node("reflector", reflector_node_partial)
workflow.add_node("roleplay", roleplay_node_partial)
workflow.add_node("planner", planner_node_partial)
workflow.add_node("executor", executor_node_partial)

workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "echo_respond": "echo_respond",
        "philosopher": "philosopher",
        "reflector": "reflector",
        "roleplay": "roleplay",
        "complex_reasoning": "planner",
    },
)

workflow.add_edge("echo_respond", END)
workflow.add_edge("philosopher", END)
workflow.add_edge("reflector", END)
workflow.add_edge("roleplay", END)
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", END)


workflow.set_entry_point("router")
app = workflow.compile(store=store)

# Generate and save the graph visualization
# Note: You may need to install pygraphviz for this to work:
# pip install pygraphviz
with open("mermaid.png", "wb") as f:
    f.write(app.get_graph(xray=True).draw_mermaid_png())


config = {"configurable": {"user_id": "Lily"}}

initial_state = AgentState(message=incoming_message, classification=None, reasoning=None, response=None, memory=None, scratchpad=None, roleplay_count=0)

# First, run the main conversational graph to get the agent's response
print("Invoking conversational graph...")
result = app.invoke(initial_state, config=config) # type: ignore
pprint.pprint(result)
print("Graph finished.")


# Now, invoke the memory manager on the full interaction
# We create a list of messages including the user's input and the AI's response
print("\nInvoking memory_manager on the completed conversation...")
try:
    # The memory manager needs a list of messages to process
    conversation_messages = [
        HumanMessage(content=result['message']),
        # The agent's response is also a message! Add it to the list.
        # You might need to adjust the role depending on your LLM, but 'assistant' or 'ai' is common.
        # For simplicity, we can treat it as another HumanMessage for the memory extractor.
        HumanMessage(content=result['response'])
    ]
    memory_manager.invoke(
        {"messages": [HumanMessage(content=incoming_message)], "max_steps": 10},
        config=config # type: ignore
    )
    print("memory_manager invoked successfully.")
except Exception as e:
    print(f"Error invoking memory_manager: {e}")


# Now, the store should contain the new memories
print("\nChecking for stored memories:")
print("Namespaces:", store.list_namespaces())
# Use pprint for cleaner output of the search result
print("Memories in collection:")
memories = store.search(('echo_star', 'Lily', 'collection'))
pprint.pprint(memories)

