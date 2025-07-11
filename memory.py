import logging
from langmem import create_memory_store_manager, create_search_memory_tool
from pydantic import BaseModel, Field
from langgraph.store.memory import InMemoryStore

class Memory(BaseModel):
    """
    A piece of information to be saved in memory.
    This could be a fact, a user preference, or a summary of the interaction.
    """
    content: str = Field(description="The content of the memory to be stored.")
    key: str = Field(description="The key of the memory to be stored.")
    value: str = Field(description="The value of the memory to be stored.")

store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)
# This manager will run in the background to extract and save memories.
# We are giving it instructions on what to look for in the conversation.
memory_manager = create_memory_store_manager(
    "openai:gpt-4.1-mini",
    namespace=("echo_star", "Lily", "collection"),
    schemas=[Memory],
    instructions="Extract key facts, user preferences, or important topics from the conversation to store for future reference. Focus on information that will help in future interactions.",
    store=store
)
logging.info("Memory Manager created successfully.")


logging.basicConfig(level=logging.INFO)


logging.info(f"Memory Manager Model Name: {getattr(memory_manager, 'model_name', getattr(memory_manager, 'model', 'Unknown'))}")


search_memory_tool = create_search_memory_tool(
    namespace=("echo_star", "Lily", "collection")
)


logging.info(f"Search Memory Tool Args: {search_memory_tool.args}")