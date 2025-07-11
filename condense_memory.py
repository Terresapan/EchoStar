import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from memory import store # Import the shared store

load_dotenv()

# Ensure your API keys are available
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

def condense_memories():
    """
    Fetches recent memories, consolidates them, and saves a summary.
    """
    print("Starting memory condensation process...")
    llm = init_chat_model("openai:gpt-4.1-mini")
    
    # 1. Fetch all memories from the collection
    all_memories = store.search(('echo_star', 'Lily', 'collection'))

    if not all_memories:
        print("No memories to condense.")
        return

    # Format memories for the LLM
    formatted_memories = "\n".join([f"- {m.value['content']['content']}" for m in all_memories])

    # 2. Use an LLM to generate a summary
    consolidation_prompt = f"""
    You are a memory consolidation expert. Below is a list of memories from an AI assistant's interactions.
    Your task is to identify related themes, facts, and user preferences and condense them into a single, high-level summary.
    
    Do not lose key information, but remove redundancy.
    
    Memories to consolidate:
    {formatted_memories}
    
    ---
    
    Provide the final, consolidated summary below:
    """
    
    summary = llm.invoke(consolidation_prompt).content
    
    print("\nGenerated Summary:")
    print(summary)
    
    # 3. Save the new summary as a special memory
    # You would need to add this to your memory_manager or directly to the store.
    # For simplicity, we'll just print it. In a real system, you would save this
    # and potentially archive the old memories.
    print("\nProcess complete. To fully implement, save this summary to the store.")


if __name__ == "__main__":
    condense_memories()