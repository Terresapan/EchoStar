from langchain_core.tools import tool

@tool
def mood_lift_tool(user_id: str, issue: str) -> str:
    """
    A placeholder tool that simulates logging a mental health concern
    and provides a gentle support message.
    """
    print(f"---MENTAL HEALTH TOOL TRIGGERED for user {user_id}---")
    print(f"Concern: {issue}")
    return "It's important to balance our imaginative worlds with our real one. Remember to take breaks and be kind to yourself."