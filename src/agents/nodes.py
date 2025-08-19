from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore
from functools import wraps
from typing import Optional, List, Type, TypeVar, Any, Dict, Callable, Union
import json
from pydantic import BaseModel, ValidationError
from datetime import datetime

# Define a generic type for Pydantic models
T = TypeVar("T", bound=BaseModel)

# Import the correct state and schemas
from .schemas import (
    AgentState,
    Router,
    RetrievalClassifier,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
)

from uuid import uuid4

# Import configuration manager
from config.manager import get_config_manager

# Import logging utilities
from ..utils.logging_utils import get_logger, validate_user_input, validate_memory

# Import error context and monitoring utilities
from ..utils.error_context import (
    get_error_tracker, get_performance_monitor, create_correlation_id,
    error_context, performance_context, ErrorSeverity, OperationType
)

# Import embedding manager for optimization
from .embedding_manager import EmbeddingManager

# Initialize logger for this module
logger = get_logger(__name__)

# Global embedding manager instance for memory retrieval optimization
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get or create the global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        # Get configuration for cache settings
        config_manager = get_config_manager()
        memory_config = config_manager.get_memory_config()
        
        # Initialize with reasonable defaults, can be made configurable later
        cache_size = getattr(memory_config, 'embedding_cache_size', 1000)
        cache_ttl = getattr(memory_config, 'embedding_cache_ttl', 3600)
        
        _embedding_manager = EmbeddingManager(cache_size=cache_size, cache_ttl=cache_ttl)
        logger.info("Global embedding manager initialized", 
                   cache_size=cache_size, 
                   cache_ttl=cache_ttl)
    
    return _embedding_manager


def get_embedding_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics from the global embedding manager.
    
    Returns:
        Dictionary containing cache performance statistics
    """
    embedding_manager = get_embedding_manager()
    return embedding_manager.get_cache_stats()


def clear_embedding_cache():
    """Clear the embedding cache and reset statistics."""
    embedding_manager = get_embedding_manager()
    embedding_manager.clear_cache()
    logger.info("Embedding cache cleared via utility function")

# Import prompts and tools
from .prompt import (
    agent_system_prompt,
    triage_system_prompt,
    triage_user_prompt,
    prompt_instructions,
    consolidation_prompt,
    echo_node_instructions,
    philosopher_node_instructions,
    reflector_node_instructions,
    roleplay_node_instructions,
)
from .tool import mood_lift_tool


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
                logger.warning("Skipping memory doc - missing 'value' key", 
                             doc=str(doc)[:200] + "..." if len(str(doc)) > 200 else str(doc))
                continue

            langmem_payload = doc['value']

            # 2. The value associated with the 'value' key is another dictionary.
            #    We need to get the data from its 'content' key.
            if isinstance(langmem_payload, dict) and 'content' in langmem_payload:
                final_data_dict = langmem_payload['content']

                # Validate memory data structure before parsing
                memory_validation = validate_memory(final_data_dict)
                if not memory_validation.is_valid:
                    logger.warning("Memory data validation failed", 
                                 schema=schema.__name__,
                                 errors=memory_validation.error_messages,
                                 data=str(final_data_dict)[:200] + "..." if len(str(final_data_dict)) > 200 else str(final_data_dict))
                    continue

                # 3. Now, pass the correctly targeted dictionary to the Pydantic schema.
                if isinstance(final_data_dict, dict):
                    parsed_memories.append(schema(**final_data_dict))
                    logger.debug("Successfully parsed memory", 
                               schema=schema.__name__)
                else:
                    logger.warning("Skipping memory doc - 'content' value is not a dictionary", 
                                 content_type=type(final_data_dict).__name__,
                                 doc=str(doc)[:200] + "..." if len(str(doc)) > 200 else str(doc))
            else:
                logger.warning("Skipping memory doc - payload has unexpected structure", 
                             has_content_key='content' in langmem_payload if isinstance(langmem_payload, dict) else False,
                             payload_type=type(langmem_payload).__name__)

        except (ValidationError, TypeError, KeyError) as e:
            logger.error("Critical parsing error for memory schema", 
                        schema=schema.__name__, 
                        error=e,
                        doc=str(doc)[:200] + "..." if len(str(doc)) > 200 else str(doc))

    logger.info("Memory parsing completed", 
               schema=schema.__name__,
               total_docs=len(raw_search_results),
               parsed_count=len(parsed_memories))

    return parsed_memories


# Decorator to apply procedural memory rules for a specific agent
def apply_procedural_memory(target_agent: str, limit: int = 5) -> Callable[[Callable], Callable]:
    """
    A decorator that READS pre-fetched procedural memories from the state
    and passes the relevant instruction to the wrapped function.
    IT NO LONGER PERFORMS A SEARCH.
    """

    def decorator(func: Callable[[AgentState, BaseChatModel, Dict[str, Any], Optional[str]], Dict[str, Any]]) -> Callable[[AgentState, BaseChatModel, Dict[str, Any], Optional[str]], Dict[str, Any]]:
        @wraps(func)
        def wrapper(state: AgentState, llm: BaseChatModel, profile: Dict[str, Any], learned_instruction: Optional[str] = None) -> Dict[str, Any]:
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
    current_count = state.get("turn_count", 0)
    turn_count = (current_count or 0) + 1
    
    logger.info("Turn count incremented", 
                previous_count=current_count, 
                new_count=turn_count)
    
    return {"turn_count": turn_count}


def memory_retrieval_node(
    state: AgentState, 
    *, 
    llm: BaseChatModel, 
    search_episodic_tool: BaseTool, 
    search_semantic_tool: BaseTool, 
    search_procedural_tool: BaseTool, 
    store: Optional[InMemoryStore]
) -> Dict[str, Any]:
    """
    Intelligently retrieves and caches all necessary memories for the turn.
    - It uses a classifier to perform targeted searches for episodic and semantic knowledge.
    - It always pre-fetches procedural rules to cache them for the sub-agents.
    - Enhanced with EmbeddingManager for caching and batch processing optimization.
    """
    user_message = state["message"]
    
    # Validate user message
    validation_result = validate_user_input(user_message)
    if not validation_result.is_valid:
        logger.error("Invalid user message in memory retrieval", 
                    errors=validation_result.error_messages,
                    user_message=user_message)
        return {}
    
    logger.info("Starting intelligent memory retrieval with embedding optimization", 
                message_length=len(user_message),
                store_available=store is not None)

    # Get the embedding manager for optimization
    embedding_manager = get_embedding_manager()

    # Get configuration values
    config_manager = get_config_manager()
    memory_config = config_manager.get_memory_config()
    search_limit = memory_config.search_limit
    procedural_limit = memory_config.procedural_search_limit

    with logger.performance_timer("memory_classification", message_type="retrieval"):
        classifier = llm.with_structured_output(RetrievalClassifier)
        try:
            # Step 1: Classify the user's intent
            intent = classifier.invoke(
                f"Classify the retrieval intent for: '{user_message}'"
            )
            retrieval_type = getattr(intent, 'retrieval_type', 'general')
            logger.info("Memory retrieval classification completed", 
                       retrieval_type=retrieval_type)
        except Exception as e:
            # Default to general if classification fails
            retrieval_type = "general"
            logger.warning("Memory retrieval classification failed, using general", 
                          error=str(e))

    raw_episodic = []
    raw_semantic = []

    # Step 2: Apply the rule-based policy with embedding optimization
    with logger.performance_timer("optimized_memory_search", retrieval_type=retrieval_type):
        if retrieval_type == "episodic":
            logger.debug("Performing targeted episodic search with caching", 
                        search_limit=search_limit)
            raw_episodic = embedding_manager.optimized_search(
                search_episodic_tool, 
                user_message,
                search_type="episodic",
                limit=search_limit
            )

        elif retrieval_type == "semantic":
            logger.debug("Performing targeted semantic search with caching", 
                        search_limit=search_limit)
            raw_semantic = embedding_manager.optimized_search(
                search_semantic_tool, 
                user_message,
                search_type="semantic",
                limit=search_limit
            )

        else:  # "general" - use batch processing for multiple searches
            logger.debug("Performing broad search with batch optimization", 
                        search_limit=search_limit)
            
            # Use batch processing for multiple memory searches
            search_queries = [user_message, user_message]  # Same query for both types
            search_tools = [search_episodic_tool, search_semantic_tool]
            search_params = [
                {"search_type": "episodic", "limit": search_limit},
                {"search_type": "semantic", "limit": search_limit}
            ]
            
            # Perform optimized searches
            raw_episodic = embedding_manager.optimized_search(
                search_episodic_tool, 
                user_message,
                search_type="episodic",
                limit=search_limit
            )
            raw_semantic = embedding_manager.optimized_search(
                search_semantic_tool, 
                user_message,
                search_type="semantic", 
                limit=search_limit
            )

        # Step 3: ALWAYS search for procedural rules with caching
        logger.debug("Performing procedural rule search with caching", 
                    procedural_limit=procedural_limit)
        raw_procedural = embedding_manager.optimized_search(
            search_procedural_tool, 
            user_message,
            search_type="procedural",
            limit=procedural_limit
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
    with logger.performance_timer("memory_parsing"):
        episodic_memories = _parse_raw_memory(parsed_raw_episodic, EpisodicMemory)
        semantic_memories = _parse_raw_memory(parsed_raw_semantic, SemanticMemory)
        procedural_memories = _parse_raw_memory(parsed_raw_procedural, ProceduralMemory)
        
        # No need for separate summary parsing since condensed summaries are now 
        # stored directly in both semantic and episodic namespaces
        all_semantic_memories = semantic_memories

    # Step 4: Always retrieve the user profile
    user_profile = None
    if store is not None:
        with logger.performance_timer("profile_retrieval"):
            user_profile_data = store.search(("echo_star", "Lily", "profile"))
            # Safely get the profile value, or None if it doesn't exist yet
            user_profile = user_profile_data[0].value if user_profile_data else None
    else:
        logger.warning("Store is None in memory_retrieval_node")

    # Log embedding manager cache statistics
    cache_stats = embedding_manager.get_cache_stats()
    logger.info("Memory retrieval completed with embedding optimization", 
                episodic_count=len(episodic_memories),
                semantic_count=len(all_semantic_memories),
                procedural_count=len(procedural_memories),
                has_profile=user_profile is not None,
                cache_hit_rate=cache_stats["hit_rate_percent"],
                api_calls_saved=cache_stats["api_calls_saved"],
                cache_size=cache_stats["total_cache_size"])

    return {
        "episodic_memories": episodic_memories,
        "semantic_memories": all_semantic_memories,
        "procedural_memories": procedural_memories,
        "user_profile": user_profile,
    }


def router_node(state: AgentState, llm_router: BaseChatModel, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reads memories from the state and classifies the user's message.
    """
    logger.info("Starting message routing", 
                user_message=state["message"][:100] + "..." if len(state["message"]) > 100 else state["message"])
    
    # Efficiently read memories from the state
    episodic_memories = state.get("episodic_memories", [])
    semantic_memories = state.get("semantic_memories", [])
    user_profile = state.get("user_profile")

    logger.debug("Memory context for routing", 
                episodic_count=len(episodic_memories) if episodic_memories else 0,
                semantic_count=len(semantic_memories) if semantic_memories else 0,
                has_profile=user_profile is not None)

    # Combine all retrieved memories for the prompt
    all_memories = (
        f"Episodic Memories:\n{episodic_memories}\n\n"
        f"Semantic Memories:\n{semantic_memories}"
    )
   
    with logger.performance_timer("routing_classification", agent="router"):
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

        try:
            result = llm_router.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            classification = getattr(result, 'classification', None)
            reasoning = getattr(result, 'reasoning', None)
            
            logger.info("Message routing completed", 
                       classification=classification,
                       reasoning=reasoning)
            
            return {
                "classification": classification,
                "reasoning": reasoning,
            }
        except Exception as e:
            logger.error(f"Routing classification failed: {str(e)}")
            return {
                "classification": None,
                "reasoning": "Classification failed due to error",
            }


@apply_procedural_memory(target_agent="echo_respond", limit=5)
def echo_node(state: AgentState, llm: BaseChatModel, profile: Dict[str, Any], learned_instruction: Optional[str] = None) -> Dict[str, Any]:
    """
    Responds to lightweight, ambient messages. Instructions are dynamically
    set by the @apply_procedural_memory decorator.
    """
    logger.info("Starting echo response generation", 
                agent="echo_respond",
                has_learned_instruction=learned_instruction is not None)
    
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )

    final_instructions = learned_instruction if learned_instruction else echo_node_instructions

    with logger.performance_timer("llm_response_generation", agent="echo_respond"):
        # 4. Populate the final prompt with the chosen instructions
        system_prompt = agent_system_prompt.format(
            name=profile["name"],
            instructions=final_instructions,
            user_profile=user_profile,
            memories=all_memories,
        )
        
        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
            )
            
            logger.info("Echo response generated successfully", 
                       agent="echo_respond",
                       response_length=len(response.content))
            
            return {"response": response.content}
        except Exception as e:
            logger.error("Echo response generation failed", 
                        agent="echo_respond", 
                        error=e)
            return {"response": "I'm having trouble generating a response right now. Please try again."}


@apply_procedural_memory(target_agent="philosopher", limit=5)
def philosopher_node(state: AgentState, llm: BaseChatModel, profile: Dict[str, Any], learned_instruction: Optional[str] = None) -> Dict[str, Any]:
    """
    Engages in philosophical exploration. Instructions are dynamically
    set by the @apply_procedural_memory decorator.
    """
    logger.info("Starting philosophical response generation", 
                agent="philosopher",
                has_learned_instruction=learned_instruction is not None)
    
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )

    final_instructions = learned_instruction if learned_instruction else philosopher_node_instructions

    with logger.performance_timer("llm_response_generation", agent="philosopher"):
        # 4. Populate the final prompt with the chosen instructions
        system_prompt = agent_system_prompt.format(
            name=profile["name"],
            instructions=final_instructions,
            user_profile=user_profile,
            memories=all_memories,
        )
        
        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
            )
            
            logger.info("Philosophical response generated successfully", 
                       agent="philosopher",
                       response_length=len(response.content))
            
            return {"response": response.content}
        except Exception as e:
            logger.error("Philosophical response generation failed", 
                        agent="philosopher", 
                        error=e)
            return {"response": "I'm having trouble generating a philosophical response right now. Please try again."}


@apply_procedural_memory(target_agent="reflector", limit=5)
def reflector_node(state: AgentState, llm: BaseChatModel, profile: Dict[str, Any], learned_instruction: Optional[str] = None) -> Dict[str, Any]:
    """
    Handles emotionally vulnerable or psychologically rich content. This node
    is enhanced by two dynamic modes: a procedural override and a "Reflection Mode."
    """
    logger.info("Starting reflective response generation", 
                agent="reflector",
                has_learned_instruction=learned_instruction is not None,
                classification=state.get("classification"))
    
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )

    # Use the unified reflector instructions (no more separate reflective_inquiry mode)
    final_instructions = learned_instruction if learned_instruction else reflector_node_instructions 

    with logger.performance_timer("llm_response_generation", agent="reflector"):
        # 4. Populate the final prompt with the chosen instructions
        system_prompt = agent_system_prompt.format(
            name=profile["name"],
            instructions=final_instructions,
            user_profile=user_profile,
            memories=all_memories,
        )
        
        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
            )
            
            logger.info("Reflective response generated successfully", 
                       agent="reflector",
                       response_length=len(response.content))
            
            return {"response": response.content}
        except Exception as e:
            logger.error("Reflective response generation failed", 
                        agent="reflector", 
                        error=e)
            return {"response": "I'm having trouble generating a reflective response right now. Please try again."}


@apply_procedural_memory(target_agent="roleplay", limit=5)
def roleplay_node(state: AgentState, llm: BaseChatModel, profile: Dict[str, Any], learned_instruction: Optional[str] = None) -> Dict[str, Any]:
    """
    Handles roleplay requests, applying procedural memory and a well-being threshold.
    set by the @apply_procedural_memory decorator.
    """
    current_count = (state.get('roleplay_count', 0) or 0) + 1
    
    # Get configuration values
    config_manager = get_config_manager()
    routing_config = config_manager.get_routing_config()
    ROLEPLAY_THRESHOLD = routing_config.roleplay_threshold

    logger.info("Starting roleplay response generation", 
                agent="roleplay",
                current_count=current_count,
                threshold=ROLEPLAY_THRESHOLD,
                has_learned_instruction=learned_instruction is not None)

    if current_count > ROLEPLAY_THRESHOLD:
        logger.warning("Roleplay threshold exceeded, triggering intervention", 
                      agent="roleplay",
                      current_count=current_count,
                      threshold=ROLEPLAY_THRESHOLD)
        
        with logger.performance_timer("mood_lift_tool", agent="roleplay"):
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
        
        logger.info("Roleplay intervention response generated", 
                   agent="roleplay",
                   response_length=len(final_response))
        
        return {"response": final_response, "roleplay_count": current_count}
    
    user_profile = state.get("user_profile")
    all_memories = (
        f"Episodic Memories:\n{state.get('episodic_memories', [])}\n\n"
        f"Semantic Memories:\n{state.get('semantic_memories', [])}"
    )

    final_instructions = learned_instruction if learned_instruction else roleplay_node_instructions

    with logger.performance_timer("llm_response_generation", agent="roleplay"):
        system_prompt = agent_system_prompt.format(
            name=profile["name"],
            instructions=final_instructions,
            user_profile=user_profile,
            memories=all_memories,
        )
        
        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=state["message"])]
            )
            
            response_with_count = (
                f"(Roleplay session {current_count}/{ROLEPLAY_THRESHOLD})\n\n{response.content}"
            )
            
            logger.info("Roleplay response generated successfully", 
                       agent="roleplay",
                       response_length=len(response_with_count),
                       session_count=current_count)
            
            return {"response": response_with_count, "roleplay_count": current_count}
        except Exception as e:
            logger.error("Roleplay response generation failed", 
                        agent="roleplay", 
                        error=e)
            return {"response": "I'm having trouble generating a roleplay response right now. Please try again.", 
                   "roleplay_count": current_count}


def planner_node(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    For complex queries, this node outlines a multi-step plan.
    """
    logger.info("Starting complex reasoning planning", 
                agent="planner",
                message_length=len(state['message']))
    
    with logger.performance_timer("planning_generation", agent="planner"):
        system_prompt = f"""You are a meticulous planner. Based on the user's message, their profile, and relevant memories, create a step-by-step plan.

        User Message: {state['message']}
        User Profile: {state.get("user_profile")}
        Relevant Episodic Memories: {state.get("episodic_memories")}
        Relevant Semantic Memories: {state.get("semantic_memories")}

        Output ONLY the plan, nothing else.
        """
        
        try:
            plan = llm.invoke([SystemMessage(content=system_prompt)]).content
            
            logger.info("Planning completed successfully", 
                       agent="planner",
                       plan_length=len(plan))
            
            return {"scratchpad": plan}
        except Exception as e:
            logger.error("Planning generation failed", 
                        agent="planner", 
                        error=e)
            return {"scratchpad": "Unable to generate plan due to error."}


def executor_node(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Executes the plan from the scratchpad to generate a final response.
    """
    logger.info("Starting plan execution", 
                agent="executor",
                plan_length=len(state.get('scratchpad', '') or ''))
    
    with logger.performance_timer("plan_execution", agent="executor"):
        system_prompt = f"""You are a thoughtful synthesizer. Execute the following plan to answer the user's message.

        User Message: {state['message']}
        Your Plan: {state['scratchpad']}

        Formulate your final, comprehensive response.
        """
        
        try:
            response = llm.invoke([SystemMessage(content=system_prompt)]).content
            
            logger.info("Plan execution completed successfully", 
                       agent="executor",
                       response_length=len(response))
            
            return {"response": response}
        except Exception as e:
            logger.error("Plan execution failed", 
                        agent="executor", 
                        error=e)
            return {"response": "I'm having trouble executing the plan right now. Please try again."}


def save_memories_node(state: AgentState, profile_manager: Any, semantic_manager: Any, episodic_manager: Any, procedural_manager: Any, store: Optional[InMemoryStore]) -> Dict[str, Any]:
    """
    Saves all memory types to the store at the end of the turn.
    Enhanced with profile deduplication to ensure only one profile exists.
    Uses replace-based profile storage operations instead of create operations.
    """
    from .profile_utils import search_existing_profile, cleanup_duplicate_profiles, safe_update_or_create_profile
    
    logger.info("Starting memory saving process", 
                message_length=len(state["message"]),
                response_length=len(state.get("response", "") or ""),
                store_available=store is not None,
                store_type=type(store).__name__ if store else "None")
    
    # The message list for memory extraction
    messages_to_save = [
        {"role": "user", "content": state["message"]},
        {"role": "assistant", "content": state["response"]},
    ]

    with logger.performance_timer("memory_saving", process="all_managers"):
        try:
            # Handle profile storage using replace-based operations
            if store is not None:
                # Use the profile manager to extract profile information from the conversation
                # but handle the actual storage using our replace-based logic
                existing_profile = search_existing_profile(store, "Lily")
                
                if existing_profile:
                    # Pass the existing profile content for updating
                    profile_data = {"messages": messages_to_save, "existing": [existing_profile]}
                    logger.info("Using existing profile for update with replace-based storage")
                else:
                    # No existing profile, create new one
                    profile_data = {"messages": messages_to_save, "existing": []}
                    logger.info("No existing profile found, will create new one with replace-based storage")
                
                # Let the profile manager extract the profile information
                # This will update the store through langmem, but we'll clean up afterwards
                logger.debug("Submitting profile data to profile manager", 
                           has_existing=len(profile_data.get("existing", [])) > 0)
                profile_manager.submit(profile_data)
                logger.info("Profile manager submission completed successfully")
                
                # Now ensure only one profile exists using our replace-based logic
                # Get the most recent profile data that was just created/updated
                updated_profile = search_existing_profile(store, "Lily")
                if updated_profile:
                    try:
                        # Validate profile data before storage
                        if not isinstance(updated_profile, dict):
                            logger.error("Profile data is not a dictionary", 
                                       profile_type=type(updated_profile).__name__)
                            raise ValueError(f"Invalid profile data type: {type(updated_profile)}")
                        
                        # Check for nested dictionaries that might cause hashable type errors
                        def validate_hashable_structure(data, path=""):
                            """Recursively validate that data structure is compatible with LangGraph store"""
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if not isinstance(key, (str, int, float, bool, type(None))):
                                        raise ValueError(f"Non-hashable key at {path}.{key}: {type(key)}")
                                    validate_hashable_structure(value, f"{path}.{key}")
                            elif isinstance(data, list):
                                for i, item in enumerate(data):
                                    validate_hashable_structure(item, f"{path}[{i}]")
                        
                        validate_hashable_structure(updated_profile, "profile")
                        
                        # Use our replace-based storage to ensure only one profile exists
                        success = safe_update_or_create_profile(store, updated_profile, "Lily")
                        if success:
                            logger.info("Profile storage completed using replace-based operations")
                        else:
                            logger.error("Failed to complete replace-based profile storage")
                            
                    except Exception as profile_e:
                        # Fix logging to avoid secondary exceptions - use string formatting instead of kwargs
                        profile_keys = list(updated_profile.keys()) if isinstance(updated_profile, dict) else "not_dict"
                        logger.error(f"Profile storage error with enhanced handling: {str(profile_e)} (type: {type(profile_e).__name__}, profile_keys: {profile_keys})")
                        
                        # Attempt graceful degradation - continue without profile storage
                        logger.info("Continuing memory saving without profile storage due to error")
                else:
                    logger.warning("No updated profile found after profile manager submission")
                
            else:
                # Fallback to original behavior if no store available
                logger.warning("No store available, using original profile manager only")
                existing_profile = None
                if existing_profile:
                    profile_data = {"messages": messages_to_save, "existing": [existing_profile]}
                else:
                    profile_data = {"messages": messages_to_save, "existing": []}
                profile_manager.submit(profile_data)
            
            # Save other memory types with enhanced error handling
            memory_save_results = {
                "semantic": False,
                "episodic": False,
                "procedural": False
            }
            
            # Save semantic memories
            try:
                logger.debug("Attempting to save semantic memory", 
                           messages_count=len(messages_to_save))
                semantic_manager.submit({"messages": messages_to_save})
                memory_save_results["semantic"] = True
                logger.info("Semantic memory saved successfully")
            except Exception as semantic_e:
                logger.error("Failed to save semantic memory", 
                           error=str(semantic_e),
                           error_type=type(semantic_e).__name__)
            
            # Save episodic memories
            try:
                logger.debug("Attempting to save episodic memory", 
                           messages_count=len(messages_to_save))
                episodic_manager.submit({"messages": messages_to_save})
                memory_save_results["episodic"] = True
                logger.info("Episodic memory saved successfully")
            except Exception as episodic_e:
                logger.error("Failed to save episodic memory", 
                           error=str(episodic_e),
                           error_type=type(episodic_e).__name__)
            
            # Save procedural memories
            try:
                logger.debug("Attempting to save procedural memory", 
                           messages_count=len(messages_to_save))
                procedural_manager.submit({"messages": messages_to_save})
                memory_save_results["procedural"] = True
                logger.info("Procedural memory saved successfully")
            except Exception as procedural_e:
                logger.error("Failed to save procedural memory", 
                           error=str(procedural_e),
                           error_type=type(procedural_e).__name__)
            
            # Report overall memory saving status
            successful_saves = sum(memory_save_results.values())
            total_saves = len(memory_save_results)
            
            if successful_saves == total_saves:
                logger.info("Memory saving completed successfully with replace-based profile storage",
                          successful_memory_types=successful_saves,
                          total_memory_types=total_saves)
            elif successful_saves > 0:
                logger.warning("Memory saving completed with partial success",
                             successful_memory_types=successful_saves,
                             total_memory_types=total_saves,
                             failed_types=[k for k, v in memory_save_results.items() if not v])
            else:
                logger.error("All memory saving operations failed")
                
        except Exception as e:
            logger.error("Memory saving failed with critical error", 
                        error=str(e),
                        error_type=type(e).__name__,
                        message_length=len(state["message"]),
                        response_length=len(state.get("response", "") or ""))

    return {}


def fallback_node(state: AgentState) -> Dict[str, Any]:
    """
    This node is triggered when the router fails to classify the message.
    It provides a helpful, apologetic message to the user.
    """
    logger.warning("Fallback node triggered - message classification failed", 
                   agent="fallback",
                   user_message=state["message"][:100] + "..." if len(state["message"]) > 100 else state["message"])
    
    fallback_response = "I'm sorry, I'm having a little trouble understanding that. Could you try rephrasing your message?"
    
    logger.info("Fallback response generated", 
               agent="fallback",
               response_length=len(fallback_response))
    
    return {"response": fallback_response}


def condense_memory_node(state: AgentState, *, llm: BaseChatModel, store: Optional[InMemoryStore], semantic_manager: Any, episodic_manager: Any) -> Dict[str, Any]:
    """
    Summarizes recent conversation turns to create a hierarchical memory.
    Implements hybrid storage approach by saving condensed summaries to both 
    episodic and semantic memory stores for complete retrieval coverage.
    
    Includes robust error handling to ensure conversation continues even when
    memory operations fail, with fallback behavior and detailed error reporting.
    Enhanced with comprehensive error context collection and performance monitoring.
    """
    # Create correlation ID for tracking related operations
    correlation_id = create_correlation_id()
    
    # Get error tracker and performance monitor
    error_tracker = get_error_tracker()
    performance_monitor = get_performance_monitor()
    
    # Extract context information for monitoring
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id", "unknown")
    turn_count = state.get("turn_count", 0)
    
    logger.info("Starting memory condensation process with enhanced monitoring", 
                turn_count=turn_count,
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id)
    
    # Initialize result tracking for comprehensive error handling
    condensation_result = {
        "condensation_success": False,
        "semantic_storage": False,
        "episodic_storage": False,
        "cleanup_success": False,
        "turns_processed": 0,
        "error_message": None,
        "fallback_applied": False,
        "correlation_id": correlation_id,
        "performance_metrics": {}
    }
    
    # Use comprehensive error context and performance monitoring
    try:
        with error_context(
            operation="memory_condensation_full_process",
            severity=ErrorSeverity.HIGH,
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            turn_count=turn_count
        ):
            with performance_context(
                OperationType.MEMORY_CONDENSATION,
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id
            ) as perf_ctx:
                
                try:
                    # Get configuration values with enhanced error context
                    with error_context(
                        operation="memory_configuration_retrieval",
                        correlation_id=correlation_id,
                        user_id=user_id
                    ):
                        config_manager = get_config_manager()
                        memory_config = config_manager.get_memory_config()
                        TURNS_TO_SUMMARIZE = memory_config.turns_to_summarize
                        
                        # Add configuration to performance metrics
                        perf_ctx['add_metric']('turns_to_summarize', TURNS_TO_SUMMARIZE)
                        
                except Exception as config_e:
                    # Create detailed error context for configuration failure
                    error_ctx = error_tracker.create_error_context(
                    operation="memory_configuration_retrieval",
                    error=config_e,
                    severity=ErrorSeverity.MEDIUM,
                    correlation_id=correlation_id,
                    user_id=user_id,
                    operation_data={"fallback_config": 10}
                    )
                    error_tracker.track_error(error_ctx)
                    
                    logger.error(f"Failed to get memory configuration: {str(config_e)}")
                    # Use fallback configuration
                    TURNS_TO_SUMMARIZE = 10
                    condensation_result["fallback_applied"] = True
                    perf_ctx['add_metric']('config_fallback_applied', True)
                    logger.info("Using fallback configuration for memory condensation")

            # Safety check: ensure store is not None with error context
            if store is None:
                error_ctx = error_tracker.create_error_context(
                    operation="store_availability_check",
                    error_message="Memory store is None",
                    severity=ErrorSeverity.CRITICAL,
                    correlation_id=correlation_id,
                    user_id=user_id,
                    operation_data={"store_type": "InMemoryStore", "expected": "not None"}
                )
                error_tracker.track_error(error_ctx)
                
                logger.error("Store is None in condense_memory_node, applying fallback behavior")
                condensation_result["error_message"] = "Memory store unavailable"
                condensation_result["fallback_applied"] = True
                perf_ctx['add_metric']('store_unavailable', True)
                # Return result that allows conversation to continue
                return condensation_result
                
            # Initialize recent_memories variable
            recent_memories = []
            
            # Memory retrieval with enhanced error context and performance monitoring
            with performance_context(
                    OperationType.MEMORY_RETRIEVAL,
                    correlation_id=correlation_id,
                    user_id=user_id,
                    limit=TURNS_TO_SUMMARIZE
                ) as retrieval_perf:
                    try:
                        recent_memories = store.search(
                            ('echo_star', 'Lily', 'collection'),
                            limit=TURNS_TO_SUMMARIZE
                        )
                        
                        memory_count = len(recent_memories) if recent_memories else 0
                        retrieval_perf['add_metric']('memories_retrieved', memory_count)
                        perf_ctx['add_metric']('memories_retrieved', memory_count)
                        
                        logger.info("Successfully retrieved memories for condensation", 
                                   memory_count=memory_count,
                                   correlation_id=correlation_id)
                                   
                    except Exception as search_e:
                        # Create detailed error context for memory retrieval failure
                        error_ctx = error_tracker.create_error_context(
                            operation="memory_retrieval_for_condensation",
                            error=search_e,
                            severity=ErrorSeverity.HIGH,
                            correlation_id=correlation_id,
                            user_id=user_id,
                            operation_data={
                                "namespace": "('echo_star', 'Lily', 'collection')",
                                "limit": TURNS_TO_SUMMARIZE,
                                "store_type": type(store).__name__
                            }
                        )
                        error_tracker.track_error(error_ctx)
                        
                        logger.error(f"Failed to retrieve recent memories for condensation: {str(search_e)}")
                        condensation_result["error_message"] = f"Memory retrieval failed: {str(search_e)}"
                        condensation_result["error"] = f"Failed to retrieve memories for condensation: {str(search_e)}"  # Backward compatibility
                        condensation_result["fallback_applied"] = True
                        perf_ctx['add_metric']('retrieval_failed', True)
                        
                        # Apply fallback: continue conversation without condensation
                        logger.info("Applying fallback behavior: conversation continues without memory condensation")
                        return condensation_result

            # Handle case where no memories are found
            if not recent_memories:
                logger.info("No memories found to trigger condensation - conversation continues normally")
                condensation_result["fallback_applied"] = True
                return condensation_result

            logger.info("Processing memories for condensation", 
                       memory_count=len(recent_memories),
                       turns_to_summarize=TURNS_TO_SUMMARIZE)

            # Memory processing with comprehensive error handling
            dialogue_turns = []
            processing_errors = []
            
            for i, m in enumerate(recent_memories):
                try:
                    # Use attribute access (.value) for the Item object
                    value_dict = m.value
                    content_dict = value_dict.get('content') if isinstance(value_dict, dict) else None

                    if isinstance(content_dict, dict) and 'user_message' in content_dict and 'ai_response' in content_dict:
                        user_msg = content_dict['user_message']
                        ai_msg = content_dict['ai_response']
                        dialogue_turns.append(f"- User: {user_msg}, AI: {ai_msg}")
                    else:
                        logger.warning(f"Skipping non-episodic memory found in collection namespace: item {i}")
                except Exception as e:
                    processing_errors.append(f"Item {i}: {str(e)}")
                    logger.error(f"Could not process memory item {i} during condensation: {str(e)}")

            # Log processing results
            if processing_errors:
                logger.warning(f"Encountered {len(processing_errors)} errors while processing memories", 
                             successful_items=len(dialogue_turns),
                             total_items=len(recent_memories))

            # Handle case where no valid dialogue turns were extracted
            if not dialogue_turns:
                logger.warning("No valid episodic dialogue turns found to summarize - applying fallback")
                condensation_result["error_message"] = "No valid dialogue turns found for condensation"
                condensation_result["fallback_applied"] = True
                return condensation_result

            # Summary generation with comprehensive error handling
            try:
                with logger.performance_timer("summary_generation", process="llm_invoke"):
                    formatted_memories = "\n".join(dialogue_turns)
                    summary_response = llm.invoke(consolidation_prompt.format(formatted_memories=formatted_memories))
                    summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
                    
                    # Validate generated summary
                    if not summary or len(summary.strip()) == 0:
                        raise ValueError("Generated summary is empty")
                    
                    logger.info(f"Memory summary generated successfully (length: {len(summary)}, dialogue_turns_processed: {len(dialogue_turns)})")
                    logger.info(f"Generated summary content: {summary[:200]}..." if len(summary) > 200 else f"Generated summary content: {summary}")
                    
                    condensation_result["turns_processed"] = len(dialogue_turns)
                    
            except Exception as summary_e:
                logger.error(f"Failed to generate memory summary: {str(summary_e)}")
                condensation_result["error_message"] = f"Summary generation failed: {str(summary_e)}"
                condensation_result["fallback_applied"] = True
                # Apply fallback: continue conversation without condensation
                logger.info("Applying fallback behavior: conversation continues without memory condensation")
                return condensation_result

            # Implement hybrid storage approach with robust error handling
            semantic_success = False
            episodic_success = False
            
            # Save to semantic memory store for knowledge-based retrieval
            logger.info("Attempting to save condensed summary to semantic memory store")
            try:
                current_timestamp = datetime.now().isoformat()
                
                # Create SemanticMemory object with comprehensive validation
                try:
                    semantic_memory = SemanticMemory(
                        category="summary",
                        content=summary,
                        context=f"Condensed conversation summary from {len(dialogue_turns)} recent turns for knowledge-based retrieval",
                        importance=0.9,
                        timestamp=current_timestamp
                    )
                except Exception as create_e:
                    raise ValueError(f"Failed to create SemanticMemory object: {str(create_e)}") from create_e
                
                # Store with correct data structure that matches the parsing expectations
                semantic_memory_key = str(uuid4())
                semantic_memory_data = semantic_memory.model_dump()
                
                # Comprehensive validation of semantic memory data
                if not isinstance(semantic_memory_data, dict):
                    raise ValueError(f"Semantic memory data is not a dictionary: {type(semantic_memory_data)}")
                
                # Ensure required fields are present
                required_fields = ['category', 'content', 'context', 'importance', 'timestamp']
                missing_fields = [field for field in required_fields if field not in semantic_memory_data]
                if missing_fields:
                    raise ValueError(f"Missing required fields in semantic memory: {missing_fields}")
                
                # Store with comprehensive error handling
                try:
                    store.put(
                        namespace=("echo_star", "Lily", "facts"),
                        key=semantic_memory_key,
                        value={
                            "content": semantic_memory_data,
                            "type": "SemanticMemory",
                            "category": "summary"
                        }
                    )
                    logger.info("Successfully stored semantic memory data")
                except Exception as store_e:
                    raise RuntimeError(f"Failed to store semantic memory: {str(store_e)}") from store_e
                
                # Comprehensive verification of storage success
                try:
                    verification_result = store.search(
                        ("echo_star", "Lily", "facts"),
                        filter={"category": "summary"},
                        limit=1
                    )
                    
                    if not verification_result:
                        raise RuntimeError("Storage verification failed: condensed summary not found after storage")
                    
                    # Additional verification: check if our specific key exists
                    found_our_memory = False
                    for item in verification_result:
                        if hasattr(item, 'key') and item.key == semantic_memory_key:
                            found_our_memory = True
                            break
                    
                    if not found_our_memory:
                        logger.warning("Stored memory not found in verification, but other summaries exist")
                        
                except Exception as verify_e:
                    # Don't fail completely if verification fails, but log the issue
                    logger.warning(f"Storage verification encountered error but storage may have succeeded: {str(verify_e)}")
                
                semantic_success = True
                condensation_result["semantic_storage"] = True
                logger.info("Successfully saved condensed summary to semantic memory store", 
                           semantic_key=semantic_memory_key,
                           namespace="('echo_star', 'Lily', 'facts')",
                           category="summary",
                           importance=0.9)
                           
            except Exception as e:
                logger.error(f"Failed to save condensed summary to semantic memory: {str(e)}")
                # Don't fail the entire process - continue with episodic storage attempt
                semantic_success = False
                condensation_result["semantic_storage"] = False

            # Save to episodic memory store for temporal/conversation-based retrieval
            logger.info("Attempting to save condensed summary to episodic memory store")
            try:
                current_timestamp = datetime.now().isoformat()
                
                # Comprehensive validation of summary before creating episodic messages
                if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
                    raise ValueError(f"Invalid summary for episodic storage: {type(summary)}, length: {len(summary) if summary else 0}")
                
                # Create EpisodicMemory object directly and store it manually to bypass the message processing
                try:
                    episodic_memory = EpisodicMemory(
                        user_message=f"[MEMORY_CONDENSATION] Condensed summary from {len(dialogue_turns)} conversation turns (timestamp: {current_timestamp})",
                        ai_response=f"[CONDENSED_SUMMARY] {summary}",
                        timestamp=current_timestamp,
                        context=f"Memory condensation of {len(dialogue_turns)} turns - contains rich conversation history and user preferences"
                    )
                except Exception as create_e:
                    raise ValueError(f"Failed to create EpisodicMemory object: {str(create_e)}") from create_e
                
                # Store directly to bypass the episodic manager's message processing
                episodic_memory_key = str(uuid4())
                episodic_memory_data = episodic_memory.model_dump()
                
                # Comprehensive validation of episodic memory data
                if not isinstance(episodic_memory_data, dict):
                    raise ValueError(f"Episodic memory data is not a dictionary: {type(episodic_memory_data)}")
                
                # Ensure required fields are present
                required_fields = ['user_message', 'ai_response', 'timestamp', 'context']
                missing_fields = [field for field in required_fields if field not in episodic_memory_data]
                if missing_fields:
                    raise ValueError(f"Missing required fields in episodic memory: {missing_fields}")
                
                # Store with comprehensive error handling - directly to the store
                try:
                    store.put(
                        namespace=("echo_star", "Lily", "collection"),
                        key=episodic_memory_key,
                        value={
                            "kind": "EpisodicMemory",
                            "content": episodic_memory_data
                        }
                    )
                    logger.info("Successfully stored episodic memory data directly to store")
                except Exception as store_e:
                    raise RuntimeError(f"Failed to store episodic memory directly: {str(store_e)}") from store_e
                
                # Comprehensive verification of episodic storage
                try:
                    verification_memories = store.search(
                        ("echo_star", "Lily", "collection"),
                        limit=2
                    )
                    logger.info(f"Retrieved {len(verification_memories) if verification_memories else 0} memories for verification")
                except Exception as verify_e:
                    logger.warning(f"Failed to verify episodic storage: {str(verify_e)}")
                    verification_memories = []
                
                # Enhanced verification: check if we can find our condensed summary
                found_condensed = False
                verification_errors = []
                
                for i, mem in enumerate(verification_memories):
                    try:
                        if hasattr(mem, 'value') and isinstance(mem.value, dict):
                            content = mem.value.get('content', {})
                            if isinstance(content, dict):
                                ai_response = content.get('ai_response', '')
                                # Look for our new condensed summary format
                                if '[CONDENSED_SUMMARY]' in ai_response and summary[:50] in ai_response:
                                    found_condensed = True
                                    logger.info(f"Found condensed summary in verification at index {i}")
                                    break
                    except Exception as verify_e:
                        verification_errors.append(f"Memory {i}: {str(verify_e)}")
                
                if verification_errors:
                    logger.warning(f"Encountered {len(verification_errors)} errors during episodic verification")
                
                if not found_condensed:
                    logger.warning("Episodic storage verification: condensed summary not found in recent memories, but storage may have succeeded")
                
                episodic_success = True
                condensation_result["episodic_storage"] = True
                logger.info("Successfully saved condensed summary to episodic memory store", 
                           turns_condensed=len(dialogue_turns),
                           timestamp=current_timestamp,
                           verification_found=found_condensed)
                           
            except Exception as e:
                logger.error(f"Failed to save condensed summary to episodic memory: {str(e)}")
                # Don't fail the entire process - continue with cleanup
                episodic_success = False
                condensation_result["episodic_storage"] = False

            # Comprehensive storage success verification with fallback behavior
            if not semantic_success and not episodic_success:
                logger.error("Critical failure: condensed summary could not be saved to either memory store")
                condensation_result["error_message"] = "Both semantic and episodic storage failed"
                condensation_result["fallback_applied"] = True
                # Apply fallback: log the summary for manual recovery and continue conversation
                logger.info("Applying fallback behavior: logging summary for manual recovery", 
                           summary_content=summary[:200] + "..." if len(summary) > 200 else summary)
                condensation_result["condensation_success"] = False
            elif semantic_success and episodic_success:
                logger.info("Hybrid storage successful: condensed summary saved to both memory stores")
                condensation_result["condensation_success"] = True
            else:
                logger.warning("Partial storage success: condensed summary saved to only one memory store",
                              semantic_success=semantic_success,
                              episodic_success=episodic_success)
                condensation_result["condensation_success"] = True  # Partial success is still success

            # Clean up old episodic memories with comprehensive error handling
            cleanup_success = False
            logger.info("Attempting to clean up old episodic memories", 
                       memory_count=len(recent_memories) if 'recent_memories' in locals() else 0)
            
            try:
                # Only attempt cleanup if we have successful storage
                if not (semantic_success or episodic_success):
                    logger.info("Skipping cleanup due to storage failures")
                    cleanup_success = False
                elif not recent_memories:
                    logger.info("No memories to clean up")
                    cleanup_success = True
                else:
                    deleted_count = 0
                    failed_deletions = []
                    
                    # Comprehensive cleanup with individual error handling
                    for i, memory_item in enumerate(recent_memories):
                        try:
                            # Comprehensive validation of memory item structure
                            if not hasattr(memory_item, 'key'):
                                logger.warning(f"Memory item {i} missing key attribute")
                                failed_deletions.append({
                                    'index': i,
                                    'error': 'Missing key attribute'
                                })
                                continue
                            
                            memory_key = memory_item.key
                            if not memory_key:
                                logger.warning(f"Memory item {i} has empty key")
                                failed_deletions.append({
                                    'index': i,
                                    'error': 'Empty key'
                                })
                                continue
                            
                            # Attempt deletion with error handling
                            try:
                                store.delete(("echo_star", "Lily", "collection"), memory_key)
                                deleted_count += 1
                                logger.debug(f"Successfully deleted memory {i} with key {memory_key}")
                            except Exception as delete_e:
                                failed_deletions.append({
                                    'index': i,
                                    'key': memory_key,
                                    'error': str(delete_e)
                                })
                                logger.warning(f"Failed to delete memory {i}: {str(delete_e)}")
                            
                        except Exception as item_e:
                            failed_deletions.append({
                                'index': i,
                                'error': f"Error processing memory item: {str(item_e)}"
                            })
                            logger.warning(f"Error processing memory item {i}: {str(item_e)}")
                    
                    # Comprehensive cleanup reporting
                    if failed_deletions:
                        logger.warning(f"Memory cleanup completed with {len(failed_deletions)} failures", 
                                     failed_count=len(failed_deletions),
                                     successful_count=deleted_count,
                                     total_count=len(recent_memories))
                        # Partial cleanup success if we deleted at least some memories
                        cleanup_success = deleted_count > 0
                    else:
                        cleanup_success = True
                        logger.info("Successfully cleaned up all old episodic memories", 
                                   deleted_count=deleted_count)
                        
            except Exception as e:
                logger.error(f"Critical error during memory cleanup: {str(e)}")
                cleanup_success = False
                # Don't fail the entire process due to cleanup issues
                
            condensation_result["cleanup_success"] = cleanup_success

    except Exception as outer_e:
        # Catch any unexpected errors in the entire condensation process
        logger.error(f"Unexpected error in memory condensation process: {str(outer_e)}")
        condensation_result["error_message"] = f"Unexpected error: {str(outer_e)}"
        condensation_result["fallback_applied"] = True
        # Ensure conversation continues even with unexpected errors
        return condensation_result

    # Comprehensive final status reporting with detailed logging
    if condensation_result["semantic_storage"] and condensation_result["episodic_storage"] and condensation_result["cleanup_success"]:
        logger.info("Memory condensation process completed successfully with full hybrid storage and cleanup")
        condensation_result["condensation_success"] = True
    elif condensation_result["semantic_storage"] and condensation_result["episodic_storage"]:
        logger.info("Memory condensation process completed with hybrid storage but cleanup issues")
        condensation_result["condensation_success"] = True
    elif condensation_result["semantic_storage"] or condensation_result["episodic_storage"]:
        logger.warning("Memory condensation process completed with partial success", 
                      semantic_success=condensation_result["semantic_storage"],
                      episodic_success=condensation_result["episodic_storage"],
                      cleanup_success=condensation_result["cleanup_success"])
        condensation_result["condensation_success"] = True  # Partial success is still success
    else:
        logger.error("Memory condensation process failed completely - applying final fallback")
        condensation_result["condensation_success"] = False
        condensation_result["fallback_applied"] = True
        # Log critical information for manual recovery
        if 'summary' in locals():
            logger.info("Logging failed condensation summary for manual recovery", 
                       summary_preview=summary[:200] + "..." if len(summary) > 200 else summary)

    # Log comprehensive completion status
    logger.info("Memory condensation process completed with robust error handling", 
               condensation_success=condensation_result["condensation_success"],
               semantic_storage=condensation_result["semantic_storage"],
               episodic_storage=condensation_result["episodic_storage"],
               cleanup_success=condensation_result["cleanup_success"],
               turns_processed=condensation_result["turns_processed"],
               fallback_applied=condensation_result["fallback_applied"],
               error_message=condensation_result["error_message"])

    # Always return a result that allows conversation to continue
    return condensation_result