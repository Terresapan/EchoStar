import logging
import os

import streamlit as st
from langgraph.store.memory import InMemoryStore
from schemas import EpisodicMemory, SemanticMemory, UserProfile, ProceduralMemory

from langmem import create_memory_store_manager, create_search_memory_tool, ReflectionExecutor

def get_memory_store():
    """
    Initialize and return the in-memory store for EchoStar.
    This store will hold all types of memories: profile, semantic, episodic, and procedural.
    """
    print("=== STARTING MEMORY STORE INITIALIZATION ===")
   
    # --- Setup ---
    openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print(f"DEBUG: OpenAI API key set: {openai_api_key[:10]}...")

    # A single, unified store for all memory types
    try:
        store = InMemoryStore(
            index={"embed": "openai:text-embedding-3-small"}
        )
        print(f"DEBUG: Created InMemoryStore: {store}")
        print(f"DEBUG: store type: {type(store)}")
        print(f"DEBUG: store has search method: {hasattr(store, 'search')}")
        
        # Test the store immediately
        test_result = store.search(("test", "namespace"))
        print(f"DEBUG: Store search test successful: {test_result}")
        
    except Exception as e:
        print(f"ERROR creating InMemoryStore: {e}")
        print(f"Exception type: {type(e)}")
        print(f"This might be due to missing OpenAI API key or network issues")
        # Don't set store to None, raise the exception to see what's wrong
        raise e

    # --- Memory Managers (For Writing to the Store) ---
    # 1. Manager for the UserProfile (a single, evolving document)
    profile_manager = ReflectionExecutor(
        create_memory_store_manager(
        "openai:gpt-4.1-mini",
        namespace=("echo_star", "Lily", "profile"),
        schemas=[UserProfile],
        instructions=(
            "Extract and update the user's profile based on the conversation. "
            "Only update the following fields: 'name', 'background', 'communication_style', "
            "'emotional_baseline', 'inferred_attachment_style'. Do not add any other fields."
        ),
        enable_inserts=False,  # Ensures a single, evolving profile
        store=store
    ))

    # 2. Manager for SemanticMemory (a collection of individual facts)
    semantic_manager = ReflectionExecutor(
        create_memory_store_manager(
        "openai:gpt-4o-mini",
        namespace=("echo_star", "Lily", "facts"),
        schemas=[SemanticMemory],
        instructions="Extract any specific, atomic facts from the conversation, such as stated preferences, boundaries, goals, or traits.",
        enable_inserts=True,  # Correctly allows new facts to be added to the collection
        store=store,
    ))

    # 3. Manager for EpisodicMemory (a collection of conversation turns)
    episodic_manager = ReflectionExecutor(
        create_memory_store_manager(
        "openai:gpt-4o-mini",
        namespace=("echo_star", "Lily", "collection"),
        schemas=[EpisodicMemory],
        instructions="Extract the user's message and the AI's response as a single conversational turn.",
        enable_inserts=True,  # Correctly allows new episodes to be added
        store=store,
    ))

    # 4. Manager for ProceduralMemory (a collection of learned behaviors)
    procedural_manager = ReflectionExecutor(
        create_memory_store_manager(
        "openai:gpt-4o-mini",
        namespace=("echo_star", "Lily", "rules"),
        schemas=[ProceduralMemory],
        instructions="""Extract any direct user feedback that defines a behavioral rule. 
        Identify the condition (trigger) and the desired agent response (action). 
        For example, if the user says, 'If I seem sad, please use the mood lift tool,' 
        extract that as a trigger-action pair.""",
        enable_inserts=True,  # Allows new procedural rules to be added
        store=store,
    ))

    # --- Memory Search Tools (For Reading from the Store) ---
    # 1. Search tool for Episodic Memories (past conversations)
    search_episodic_tool = create_search_memory_tool(
        namespace=("echo_star", "Lily", "collection"),
        store=store,
    )

    # 2. Search tool for Semantic Memories (facts and preferences)
    search_semantic_tool = create_search_memory_tool(
        namespace=("echo_star", "Lily", "facts"),
        store=store,
    )

    # 3. Search tool for Procedural Memories (learned behaviors)
    search_procedural_tool = create_search_memory_tool(
        namespace=("echo_star", "Lily", "rules"),
        store=store,
    )

    # --- Logging ---
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "Hybrid memory system with Profile, Semantic, and Episodic managers initialized."
    )

    # Return a dictionary containing all components
    memory_dict = {
        "store": store,
        "profile_manager": profile_manager,
        "semantic_manager": semantic_manager,
        "episodic_manager": episodic_manager,
        "procedural_manager": procedural_manager,
        "search_episodic_tool": search_episodic_tool,
        "search_semantic_tool": search_semantic_tool,
        "search_procedural_tool": search_procedural_tool,
    }
    print(f"DEBUG: Returning memory_dict with store: {memory_dict['store']}")
    print("=== MEMORY STORE INITIALIZATION COMPLETE ===")
    return memory_dict