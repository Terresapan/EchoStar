from langchain_core.messages import HumanMessage, SystemMessage
from functools import wraps
from typing import Optional, List, Type, TypeVar, Any, Dict
import json
from pydantic import BaseModel, ValidationError

# Define a generic type for Pydantic models
T = TypeVar("T", bound=BaseModel)

# Import the correct state and schemas
from schemas import (
    AgentState,
    Router,
    RetrievalClassifier,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
)

from uuid import uuid4

# Import prompts and tools
from prompt import (
    agent_system_prompt,
    triage_system_prompt,
    triage_user_prompt,
    prompt_instructions,
    consolidation_prompt,
    echo_node_instructions,
    philosopher_node_instructions,
    reflector_node_instructions,
    reflective_inquiry_node_instructions,
    roleplay_node_instructions,
)
from tool import mood_lift_tool


def _parse_raw_memory(raw_search_results: List[Any], schema: Type[T]) -> List[T]:
    """
    Correctly parses a list of dictionaries returned from the InMemoryStore.
    """
    parsed_memories = []
    # raw_search_results is a list of dictionaries, where each dict is a search result.
    for doc in raw_search_results:
        try:
            # 1. Check for the 'value' key in the dictionary. This is the correct method.
            if 'value' not in doc:
                print(f"Skipping doc because dictionary has no 'value' key: {doc}")
                continue

            langmem_payload = doc['value']

            # 2. The value associated with the 'value' key is another dictionary.
            #    We need to get the data from its 'content' key.
            if isinstance(langmem_payload, dict) and 'content' in langmem_payload:
                final_data_dict = langmem_payload['content']

                # 3. Now, pass the correctly targeted dictionary to the Pydantic schema.
                if isinstance(final_data_dict, dict):
                    parsed_memories.append(schema(**final_data_dict))
                else:
                    print(f"Skipping doc because 'content' value is not a dictionary: {doc}")
            else:
                print(f"Skipping doc because payload has unexpected structure: {doc}")

        except (ValidationError, TypeError, KeyError) as e:
            print(f"CRITICAL PARSING ERROR for schema {schema.__name__}: {doc}, error: {e}")

    return parsed_memories


# Decorator to apply procedural memory rules for a specific agent
def apply_procedural_memory(target_agent: str, limit: int = 5):
    """
    A decorator that READS pre-fetched procedural memories from the state
    and passes the relevant instruction to the wrapped function.
    IT NO LONGER PERFORMS A SEARCH.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(state: AgentState, llm, profile: dict, learned_instruction: Optional[str] = None):
            print(
                f"---CHECKING FOR {target_agent.upper()}-SPECIFIC PROCEDURAL TRIGGERS (LIMIT={limit})---"
            )

            # 1. Get the pre-fetched rules from the state (the cache)
            procedural_memories = state.get("procedural_memories", [])

            if procedural_memories:
                # 2. Find a rule that specifically targets this agent or 'all'
                for rule in procedural_memories:
                    if rule.target_agent == target_agent or rule.target_agent == "all":
                        print(f"RULE APPLIED: {rule.action}")
                        learned_instruction = rule.action
                        break

            # 3. Pass the instruction to the wrapped function
            return func(
                state, llm, profile, learned_instruction=learned_instruction
            )

        return wrapper

    return decorator

def manage_turn_count(state: AgentState) -> Dict[str, Any]:
    """A node to initialize and increment the turn count."""
    # --- Start Debug ---
    current_count = state.get("turn_count", 0)
    print(f"\n[DEBUG] Entering manage_turn_count. Current count is: {current_count}")
    # --- End Debug ---

    turn_count = current_count + 1
    return {"turn_count": turn_count}


def memory_retrieval_node(state: AgentState, *, llm, search_episodic_tool, search_semantic_tool, search_procedural_tool, store) -> dict:
    """
    Intelligently retrieves and caches all necessary memories for the turn.
    - It uses a classifier to perform targeted searches for episodic and semantic knowledge.
    - It always pre-fetches procedural rules to cache them for the sub-agents.
    """
    print("---INTELLIGENT MEMORY RETRIEVAL---")
    print(f"DEBUG: store parameter in memory_retrieval_node: {store}")
    user_message = state["message"]

    classifier = llm.with_structured_output(RetrievalClassifier)
    try:
        # Step 1: Classify the user's intent
        intent = classifier.invoke(
            f"Classify the retrieval intent for: '{user_message}'"
        )
        retrieval_type = intent.retrieval_type
    except Exception:
        # Default to general if classification fails
        retrieval_type = "general"

    print(f"Retrieval Type: {retrieval_type}")

    raw_episodic = []
    raw_semantic = []

    # Step 2: Apply the rule-based policy
    if retrieval_type == "episodic":
        print("-> Performing targeted episodic search...")
        raw_episodic = search_episodic_tool.invoke(
            {"query": user_message, "limit": 10}
        )

    elif retrieval_type == "semantic":
        print("-> Performing targeted semantic search...")
        raw_semantic = search_semantic_tool.invoke(
            {"query": user_message, "limit": 10}
        )

    else:  # "general"
        print("-> Performing broad search...")
        raw_episodic = search_episodic_tool.invoke(
            {"query": user_message, "limit": 10}
        )
        raw_semantic = search_semantic_tool.invoke(
            {"query": user_message, "limit": 10}
        )

    # Step 3: ALWAYS search for procedural rules to cache them for the decorator
    print("-> Performing procedural rule search...")
    raw_procedural = search_procedural_tool.invoke(
        {"query": user_message, "limit": 5}
    )

    # CORRECTED STEP: Decode the raw tool output if it's a string.
    def _load_if_string(data: Any) -> list:
        if isinstance(data, str):
            try:
                # The entire output is a single JSON string.
                return json.loads(data)
            except json.JSONDecodeError:
                # Handle cases where the string is not valid JSON.
                print(f"Warning: Could not decode string as JSON: {data}")
                return []
        # If it's already a list, return it.
        return data if isinstance(data, list) else []

    # Pre-process each raw output before passing to the parsing utility.
    parsed_raw_episodic = _load_if_string(raw_episodic)
    parsed_raw_semantic = _load_if_string(raw_semantic)
    parsed_raw_procedural = _load_if_string(raw_procedural)

    # Now, call your robust parser with a proper list.
    episodic_memories = _parse_raw_memory(parsed_raw_episodic, EpisodicMemory)
    semantic_memories = _parse_raw_memory(parsed_raw_semantic, SemanticMemory)
    procedural_memories = _parse_raw_memory(parsed_raw_procedural, ProceduralMemory)

    # Step 4: Always retrieve the user profile
    user_profile = None
    if store is not None:
        user_profile_data = store.search(("echo_star", "Lily", "profile"))
        # Safely get the profile value, or None if it doesn't exist yet
        user_profile = user_profile_data[0].value if user_profile_data else None
    else:
        print("Warning: store is None in memory_retrieval_node")

    return {
        "episodic_memories": episodic_memories,
        "semantic_memories": semantic_memories,
        "procedural_memories": procedural_memories,
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
        user_profile_background=profile["user_profile_background"],
        triage_echo=prompt_instructions["triage_rules"]["echo_respond"],
        triage_roleplay=prompt_instructions["triage_rules"]["roleplay"],
        triage_reflector=prompt_instructions["triage_rules"]["reflector"],
        triage_philosopher=prompt_instructions["triage_rules"]["philosopher"],
        triage_complex_reasoning=prompt_instructions["triage_rules"]["complex_reasoning"],
        triage_fallback=prompt_instructions["triage_rules"]["fallback"],
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


@apply_procedural_memory(target_agent="echo_respond", limit=5)
def echo_node(state: AgentState, llm, profile: dict, learned_instruction: Optional[str] = None) -> dict:
    """
    Responds to lightweight, ambient messages. Instructions are dynamically
    set by the @apply_procedural_memory decorator.
    """
    print("---ECHO RESPOND---")
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )

    final_instructions = learned_instruction if learned_instruction else echo_node_instructions

    # 4. Populate the final prompt with the chosen instructions
    system_prompt = agent_system_prompt.format(
        name=profile["name"],
        instructions=final_instructions,
        user_profile=user_profile,
        memories=all_memories,
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
    )
    return {"response": response.content}


@apply_procedural_memory(target_agent="philosopher", limit=5)
def philosopher_node(state: AgentState, llm, profile: dict, learned_instruction: Optional[str] = None) -> dict:
    """
    Engages in philosophical exploration. Instructions are dynamically
    set by the @apply_procedural_memory decorator.
    """
    print("---PHILOSOPHER---")
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )

    final_instructions = learned_instruction if learned_instruction else philosopher_node_instructions

    # 4. Populate the final prompt with the chosen instructions
    system_prompt = agent_system_prompt.format(
        name=profile["name"],
        instructions=final_instructions,
        user_profile=user_profile,
        memories=all_memories,
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
    )
    return {"response": response.content}


@apply_procedural_memory(target_agent="reflector", limit=5)
def reflector_node(state: AgentState, llm, profile: dict, learned_instruction: Optional[str] = None) -> dict:
    """
    Handles emotionally vulnerable or psychologically rich content. This node
    is enhanced by two dynamic modes: a procedural override and a "Reflection Mode."
    """
    print("---REFLECTOR---")
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )

    # --- DYNAMIC INSTRUCTION LOGIC ---
    print("---CHECKING FOR REFLECTOR-SPECIFIC PROCEDURAL TRIGGERS---")

    base_instructions = reflector_node_instructions

    # 3. Check if Reflection Mode should be activated and append its instructions
    final_instructions = learned_instruction if learned_instruction else base_instructions
    if state.get("classification") == "reflective_inquiry":
        print("---REFLECTION MODE ACTIVATED---")
        final_instructions += f"\n\n{reflective_inquiry_node_instructions}" 

    # --- END DYNAMIC LOGIC ---

    # 4. Populate the final prompt with the chosen instructions
    system_prompt = agent_system_prompt.format(
        name=profile["name"],
        instructions=final_instructions,
        user_profile=user_profile,
        memories=all_memories,
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
    )
    return {"response": response.content}


@apply_procedural_memory(target_agent="roleplay", limit=5)
def roleplay_node(state: AgentState, llm, profile: dict, learned_instruction: Optional[str] = None) -> dict:
    """
    Handles roleplay requests, applying procedural memory and a well-being threshold.
    set by the @apply_procedural_memory decorator.
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

    final_instructions = learned_instruction if learned_instruction else roleplay_node_instructions

    system_prompt = agent_system_prompt.format(
        name=profile["name"],
        instructions=final_instructions,
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


def save_memories_node(state: AgentState, profile_manager, semantic_manager, episodic_manager, procedural_manager) -> dict:
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
    profile_manager.submit({"messages": messages_to_save, "existing": [existing_profile] if existing_profile else []})
    semantic_manager.submit({"messages": messages_to_save})
    episodic_manager.submit({"messages": messages_to_save})
    procedural_manager.submit({"messages": messages_to_save})

    return {}


def fallback_node(state: AgentState) -> dict:
    """
    This node is triggered when the router fails to classify the message.
    It provides a helpful, apologetic message to the user.
    """
    print("---FALLBACK TRIGGERED---")
    
    fallback_response = "I'm sorry, I'm having a little trouble understanding that. Could you try rephrasing your message?"
    
    return {"response": fallback_response}


def condense_memory_node(state: AgentState, *, llm, store, semantic_manager) -> dict:
    """
    Summarizes recent conversation turns to create a hierarchical memory.
    This version correctly handles Item objects from a direct store.search() call.
    """
    print("---CONDENSING RECENT MEMORIES---")
    TURNS_TO_SUMMARIZE = 10

    # Safety check: ensure store is not None
    if store is None:
        print("ERROR: Store is None in condense_memory_node. Skipping memory condensation.")
        return {}

    # This direct call returns a list of Item OBJECTS
    recent_memories = store.search(
        ('echo_star', 'Lily', 'collection'),
        limit=TURNS_TO_SUMMARIZE
    )

    if not recent_memories:
        print("No memories found to trigger condensation.")
        return {}

    dialogue_turns = []
    for m in recent_memories:
        try:
            # Use attribute access (.value) for the Item object
            value_dict = m.value
            content_dict = value_dict.get('content') if isinstance(value_dict, dict) else None

            if isinstance(content_dict, dict) and 'user_message' in content_dict and 'ai_response' in content_dict:
                user_msg = content_dict['user_message']
                ai_msg = content_dict['ai_response']
                dialogue_turns.append(f"- User: {user_msg}, AI: {ai_msg}")
            else:
                print(f"WARNING: Skipping non-episodic memory found in 'collection' namespace: {m}")
        except Exception as e:
            print(f"ERROR: Could not process memory item: {m}, Error: {e}")

    if not dialogue_turns:
        print("No valid episodic dialogue turns found to summarize.")
        return {}

    # --- The rest of the function remains the same ---
    formatted_memories = "\n".join(dialogue_turns)
    summary_response = llm.invoke(consolidation_prompt.format(formatted_memories=formatted_memories))
    summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
    print(f"---SAVING NEW SUMMARY---")
    print(summary)

    # Save the complete summary directly to semantic memory
    # We use direct store.put() instead of semantic_manager to preserve the full summary
    try:
        from uuid import uuid4
        store.put(
            ("echo_star", "Lily", "facts"),
            str(uuid4()),
            {
                "content": {
                    "category": "summary",
                    "content": summary,
                    "context": "Condensed from recent conversation history"
                }
            }
        )
        print("Successfully saved complete summary to semantic memory")
    except Exception as e:
        print(f"Failed to save summary directly to store: {e}")
        # Try the semantic manager as fallback (though it will fragment the summary)
        try:
            semantic_manager.submit({
                "messages": [
                    {"role": "user", "content": "Please save this conversation summary as a semantic memory."},
                    {"role": "assistant", "content": f"Summary: {summary}"}
                ]
            })
            print(f"Fallback: saved summary via semantic_manager (may be fragmented)")
        except Exception as fallback_error:
            print(f"Both direct store and semantic_manager failed: {fallback_error}")

    # Clean up old episodic memories after successful condensation
    try:
        print(f"Cleaning up {len(recent_memories)} old episodic memories...")
        for memory_item in recent_memories:
            # Delete each old episodic memory
            store.delete(("echo_star", "Lily", "collection"), memory_item.key)
        print("Successfully cleaned up old episodic memories")
    except Exception as e:
        print(f"Failed to clean up old memories: {e}")

    return {}