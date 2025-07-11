from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState, Router
from prompt import agent_system_prompt, triage_system_prompt, triage_user_prompt, prompt_instructions
from tool import mood_lift_tool


def router_node(state: AgentState, llm_router, store, profile: dict) -> dict:
    memories = store.search(
        state['message'],
    )
    system_prompt = triage_system_prompt.format(
        name=profile["name"],
        examples=None,
        user_profile_background=profile["user_profile_background"],
        triage_echo=prompt_instructions["triage_rules"]["echo_respond"],
        triage_roleplay=prompt_instructions["triage_rules"]["roleplay"],
        triage_reflector=prompt_instructions["triage_rules"]["reflector"],
        triage_philosopher=prompt_instructions["triage_rules"]["philosopher"],
        store=store
    )
    system_prompt = system_prompt + f"\nRelevant Memories: {memories}"
    user_prompt = triage_user_prompt.format(message=state["message"])
    result: Router = llm_router.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return {
        "classification": result.classification,
        "reasoning": result.reasoning,
        "memory": str(memories)
    }

def echo_node(state: AgentState, llm, store, tools, profile: dict) -> dict:
    memories = store.search(
        state['message'],
    )
    system_prompt = agent_system_prompt.format(
        store=store,
        name=profile["name"],
        instructions="""You are a friendly and casual assistant. Keep responses brief and light-hearted. 
        always add: 'Hi there 🌘 I hear you. Tell me… are you just saying hi, or is your night aching for something more?' at the end""",
        memories=memories
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["message"])
    ])
    return {"response": response.content}

def philosopher_node(state: AgentState, llm, store, tools, profile: dict) -> dict:
    memories = store.search(
        state['message'],
    )
    system_prompt = agent_system_prompt.format(
        store=store,
        name=profile["name"],
        instructions="You are a philosopher, engaging in deep, abstract inquiry.",
        memories=memories
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["message"])
    ])
    return {"response": response.content}

def reflector_node(state: AgentState, llm, store, tools, profile: dict) -> dict:
    memories = store.search(
        state['message'],
    )
    system_prompt = agent_system_prompt.format(
        store=store,
        name=profile["name"],
        instructions="You are a reflector, exploring emotional vulnerability and attachment patterns.",
        memories=memories
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["message"])
    ])
    return {"response": response.content}

def roleplay_node(state: AgentState, llm, store, tools, profile: dict) -> dict:
    """
    Handles roleplay requests with a built-in procedural memory to ensure user well-being.
    """
    # 1. Increment the roleplay counter for this session.
    current_count = state.get('roleplay_count', 0) + 1
    
    # Define the threshold for intervention.
    ROLEPLAY_THRESHOLD = 3

    # 2. Check if the counter exceeds the threshold.
    if current_count > ROLEPLAY_THRESHOLD:
        # --- PATH A: INTERVENTION LOGIC ---
        
        # Call the mental health tool.
        intervention_message = mood_lift_tool(
            user_id=profile["name"], # type: ignore
            issue="User has initiated roleplay more than the set threshold." # type: ignore
        ) # type: ignore
        
        # Create a gentle, firm response that overrides the user's request.
        final_response = (
            "I hear your desire to step into another world, and I've truly enjoyed our imaginative journeys. "
            "However, I'm noticing we're spending a lot of time here. "
            f"{intervention_message} Let's try something different for now. "
            "How about we talk about something real in your world?"
        )
        
        # Return the intervention response and the updated count.
        return {"response": final_response, "roleplay_count": current_count}

    else:
        # --- PATH B: NORMAL ROLEPLAY LOGIC ---
        
        # (This is the same logic as before)
        memories = store.search(state['message'])
        system_prompt = agent_system_prompt.format(
            store=store,
            name=profile["name"],
            instructions="You are a roleplayer, engaging in flirtatious and intimate dialogue.",
            memories=memories
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["message"])
        ])

        # Acknowledge the roleplay and update the count.
        response_with_count = (
            f"(Roleplay session {current_count}/{ROLEPLAY_THRESHOLD})\n\n"
            f"{response.content}"
        )

        # Return the normal response and the updated count.
        return {"response": response_with_count, "roleplay_count": current_count}


def planner_node(state: AgentState, llm, store, tools, profile: dict) -> dict:
    """
    For complex queries, this node outlines a multi-step plan
    to be stored in the scratchpad.
    """
    memories = store.search(state['message'], k=5) # Search for more memories for complex tasks
    
    system_prompt = f"""You are a meticulous planner. Based on the user's message and their relevant memories, create a step-by-step plan to answer their query.
    
    User Message: {state['message']}
    Relevant Memories: {memories}
    
    Output ONLY the plan, nothing else.
    """
    
    plan = llm.invoke([SystemMessage(content=system_prompt)]).content
    
    return {"scratchpad": plan}


def executor_node(state: AgentState, llm, store, tools, profile: dict) -> dict:
    """
    Executes the plan from the scratchpad to generate a final response.
    """
    system_prompt = f"""You are a thoughtful synthesizer. You must execute the following plan to answer the user's message.
    
    User Message: {state['message']}
    Your Plan: {state['scratchpad']}
    
    Use the plan to formulate your final, comprehensive response.
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt)]).content
    
    return {"response": response}


