import logging
import os

import streamlit as st
from langgraph.store.memory import InMemoryStore
# Make sure schemas are imported correctly
from schemas import EpisodicMemory, SemanticMemory, UserProfile

from langmem import create_memory_store_manager, create_search_memory_tool

# --- Setup ---
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# A single, unified store for all memory types
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# --- Memory Managers (For Writing to the Store) ---
# 1. Manager for the UserProfile (a single, evolving document)
profile_manager = create_memory_store_manager(
    "openai:gpt-4.1-mini",
    namespace=("echo_star", "Lily", "profile"),
    schemas=[UserProfile],
    instructions="Extract and update the user's profile based on the conversation. Focus on preferences, emotional state, and communication style.",
    enable_inserts=False,  # Ensures a single, evolving profile
    store=store
)

# 2. Manager for SemanticMemory (a collection of individual facts)
semantic_manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    namespace=("echo_star", "Lily", "facts"),
    schemas=[SemanticMemory],
    instructions="Extract any specific, atomic facts from the conversation, such as stated preferences, boundaries, goals, or traits.",
    enable_inserts=True,  # Correctly allows new facts to be added to the collection
    store=store,
)

# 3. Manager for EpisodicMemory (a collection of conversation turns)
episodic_manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    namespace=("echo_star", "Lily", "collection"),
    schemas=[EpisodicMemory],
    instructions="Extract the user's message and the AI's response as a single conversational turn.",
    enable_inserts=True,  # Correctly allows new episodes to be added
    store=store,
)

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

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logging.info(
    "Hybrid memory system with Profile, Semantic, and Episodic managers initialized."
)
