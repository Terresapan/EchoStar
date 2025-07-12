from langchain_core.messages import HumanMessage, SystemMessage

# Import the correct state and schemas
from schemas import AgentState, Router

# Import prompts and tools
from prompt import (
    agent_system_prompt,
    triage_system_prompt,
    triage_user_prompt,
    prompt_instructions,
)
from tool import mood_lift_tool

# Import all the necessary memory components
from memory import (
    store,
    profile_manager,
    semantic_manager,
    episodic_manager,
    search_episodic_tool,
    search_semantic_tool,
)


def memory_retrieval_node(state: AgentState) -> dict:
    """
    Fetches the user profile, relevant semantic facts, and recent episodic
    memories at the start of the graph to avoid redundant lookups.
    """
    print("---RETRIEVING MEMORIES---")
    user_message = state["message"]

    # 1. Retrieve Episodic Memories (past conversations)
    episodic_memories = search_episodic_tool.invoke({"query": user_message})

    # 2. Retrieve Semantic Memories (facts, preferences, boundaries)
    semantic_memories = search_semantic_tool.invoke({"query": user_message})

    # 3. Retrieve the User Profile (the single, evolving document)
    user_profile_data = store.search(("echo_star", "Lily", "profile"))
    # Safely get the profile value, or None if it doesn't exist yet
    user_profile = user_profile_data[0].value if user_profile_data else None

    return {
        "episodic_memories": episodic_memories,
        "semantic_memories": semantic_memories,
        "user_profile": user_profile,
    }


def router_node(state: AgentState, llm_router, profile: dict) -> dict:
    """
    Reads memories from the state and classifies the user's message.
    """
    print("---ROUTING---")
    # Efficiently read memories from the state
    episodic_memories = state.get("episodic_memories", [])
    semantic_memories = state.get("semantic_memories", [])
    user_profile = state.get("user_profile")

    # Combine all retrieved memories for the prompt
    all_memories = (
        f"Episodic Memories:\n{episodic_memories}\n\n"
        f"Semantic Memories:\n{semantic_memories}"
    )
   
    system_prompt = triage_system_prompt.format(
        name=profile["name"],
        examples=None,
        user_profile_background=profile["user_profile_background"],
        triage_echo=prompt_instructions["triage_rules"]["echo_respond"],
        triage_roleplay=prompt_instructions["triage_rules"]["roleplay"],
        triage_reflector=prompt_instructions["triage_rules"]["reflector"],
        triage_philosopher=prompt_instructions["triage_rules"]["philosopher"],
    )
    system_prompt = system_prompt + f"\nRelevant Memories: {all_memories}\nUser Profile: {user_profile}"
    user_prompt = triage_user_prompt.format(message=state["message"])

    result: Router = llm_router.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return {
        "classification": result.classification,
        "reasoning": result.reasoning,
    }

def echo_node(state: AgentState, llm, profile: dict) -> dict:
    print("---ECHO RESPOND---")
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )

    system_prompt = agent_system_prompt.format(
        name=profile["name"],
        instructions="""You are a friendly and casual assistant. Weave the user's profile and past interactions into your response naturally.""",
        user_profile=user_profile,
        memories=all_memories,
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
    )
    return {"response": response.content}

def philosopher_node(state: AgentState, llm, profile: dict) -> dict:
    print("---PHILOSOPHER---")
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )
    system_prompt = agent_system_prompt.format(
        name=profile["name"],
        instructions="""You are a philosopher. Synthesize the user's profile and past memories to inform your philosophical exploration.""",
        user_profile=user_profile,
        memories=all_memories,
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
    )
    return {"response": response.content}

def reflector_node(state: AgentState, llm, profile: dict) -> dict:
    print("---REFLECTOR---")
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )
    system_prompt = agent_system_prompt.format(
        name=profile["name"],
        instructions="""You are a reflector. Use the user's profile and past memories to guide your reflection on emotional vulnerability.""",
        user_profile=user_profile,
        memories=all_memories,
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
    )
    return {"response": response.content}

def roleplay_node(state: AgentState, llm, profile: dict) -> dict:
    """
    Handles roleplay requests with a built-in procedural memory to ensure user well-being.
    """
    print("---ROLEPLAY---")
    current_count = state.get('roleplay_count', 0) + 1
    ROLEPLAY_THRESHOLD = 2

    if current_count > ROLEPLAY_THRESHOLD:
        intervention_message = mood_lift_tool.invoke({
            "user_id": profile["name"], 
            "issue": "User has initiated roleplay more than the set threshold."
        })
        final_response = (
            "I hear your desire to step into another world, and I've truly enjoyed our imaginative journeys. "
            "However, I'm noticing we're spending a lot of time here. "
            f"{intervention_message} Let's try something different for now. "
            "How about we talk about something real in your world?"
        )
        return {"response": final_response, "roleplay_count": current_count}
    
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )
    system_prompt = agent_system_prompt.format(
        name=profile["name"],
        instructions="""You are a master roleplayer. Fully embody the requested persona. Subtly incorporate details from the user's profile and memories.""",
        user_profile=user_profile,
        memories=all_memories,
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
    )
    response_with_count = (
        f"(Roleplay session {current_count}/{ROLEPLAY_THRESHOLD})\n\n{response.content}"
    )
    return {"response": response_with_count, "roleplay_count": current_count}

def planner_node(state: AgentState, llm) -> dict:
    """
    For complex queries, this node outlines a multi-step plan.
    """
    print("---PLANNING---")
    system_prompt = f"""You are a meticulous planner. Based on the user's message, their profile, and relevant memories, create a step-by-step plan.

    User Message: {state['message']}
    User Profile: {state.get("user_profile")}
    Relevant Episodic Memories: {state.get("episodic_memories")}
    Relevant Semantic Memories: {state.get("semantic_memories")}

    Output ONLY the plan, nothing else.
    """
    plan = llm.invoke([SystemMessage(content=system_prompt)]).content
    return {"scratchpad": plan}


def executor_node(state: AgentState, llm) -> dict:
    """
    Executes the plan from the scratchpad to generate a final response.
    """
    print("---EXECUTING---")
    system_prompt = f"""You are a thoughtful synthesizer. Execute the following plan to answer the user's message.

    User Message: {state['message']}
    Your Plan: {state['scratchpad']}

    Formulate your final, comprehensive response.
    """
    response = llm.invoke([SystemMessage(content=system_prompt)]).content
    return {"response": response}


def save_memories_node(state: AgentState) -> dict:
    """
    Saves all memory types to the store at the end of the turn.
    """
    print("---SAVING MEMORIES---")
    # The message list for memory extraction
    messages_to_save = [
        {"role": "user", "content": state["message"]},
        {"role": "assistant", "content": state["response"]},
    ]

    # Get the existing profile from the state
    existing_profile = state.get("user_profile")

    # Invoke all three managers to write to the store
    profile_manager.invoke({"messages": messages_to_save, "existing": [existing_profile] if existing_profile else []}) # type: ignore
    semantic_manager.invoke({"messages": messages_to_save}) # type: ignore
    episodic_manager.invoke({"messages": messages_to_save}) # type: ignore

    return {}