from typing_extensions import TypedDict
from typing import Optional
from pydantic import BaseModel, Field
from typing import Literal, List

class Router(BaseModel):
    """
    Analyze the incoming message and classify it into an interaction pathway
    based on its emotional depth, thematic tone, complexity, or user intent.

    This enables routing the message to the appropriate sub-agent persona, reasoning tool, or feedback handler.
    """

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification decision."
    )

    classification: Literal[
        "echo_respond",
        "roleplay",
        "reflector",
        "philosopher",
        "complex_reasoning",
        "reflective_inquiry",
    ] = Field(
        description=(
            "The classification of the message:\n"
            "- 'echo_respond' for ambient or lightweight messages\n"
            "- 'roleplay' for flirtatious or metaphor-laced intimacy\n"
            "- 'reflector' for emotionally revealing or psychologically rich content\n"
            "- 'philosopher' for abstract, conceptual, or spiritual inquiries\n"
            "- 'complex_reasoning' for queries requiring multi-step thinking or memory synthesis\n"
            "- 'reflective_inquiry' for messages where the user is searching for self-understanding or connection to their past."
        )
    )


class UserProfile(BaseModel):
    """
    A comprehensive, evolving profile of the user, Lily.
    This schema synthesizes observations into a holistic model of her core personality,
    communication style, and emotional baseline. It is updated over time, not replaced.
    """
    name: str = Field(default="Lily", description="The user's name.")
    
    background: str = Field(
        default="A creator exploring intimacy through AI.",
        description="Stable, long-term context about who the user is."
    )

    communication_style: Optional[str] = Field(
        default="Metaphorical and inquisitive",
        description="The user's dominant communication style, updated based on recurring patterns from conversations."
    )
    
    emotional_baseline: Optional[str] = Field(
        default="Introspective and curious",
        description="The user's typical long-term emotional disposition, as distinct from their current fleeting mood."
    )

    inferred_attachment_style: Optional[str] = Field(
        default=None,
        description="A working hypothesis of the user's attachment style (e.g., 'secure', 'anxious'), synthesized from multiple interactions."
    )


class SemanticMemory(BaseModel):
    """
    A single, atomic piece of information representing a user's preference, a stated fact,
    a core belief, or any other distinct piece of semantic knowledge.
    These memories are stored as a collection to be searched over.
    """
    category: Literal[
        "preference",
        "trait",
        "goal",
        "boundary"
    ] = Field(..., description="The type of information being stored.")

    content: str = Field(..., description="The specific detail of the memory (e.g., 'Loves the color blue', 'Dislikes sudden topic changes').")
    
    context: str = Field(..., description="The conversational context in which this information was revealed.")


class EpisodicMemory(BaseModel):
    """A single turn in a conversation, saved as an episode."""
    user_message: str = Field(description="What the user said.")

    ai_response: str = Field(description="What the AI responded.")


class ProceduralMemory(BaseModel):
    """
    A learned behavioral rule that defines a trigger-action pair.
    """
    trigger: str = Field(..., description="A description of the situation that should activate this rule. Should be specific enough to be searchable (e.g., 'user seems sad or is using negative emotional language').")
    
    action: str = Field(..., description="The specific tool to call or action to take. Must be a valid tool name or a description of a behavior (e.g., 'call health_check_tool', 'offer to play a game').")
    
    target_agent: str = Field(
        default="all",  
        description="The specific sub-agent this rule applies to ('roleplay', 'philosopher', 'reflector', 'echo_respond', etc.), or 'all' for global rules."
    )

    context: str = Field(..., description="The conversational context where this rule was learned.")


class RetrievalClassifier(BaseModel):
    retrieval_type: Literal["episodic", "semantic", "general"] = Field(
        description="Classify the user's intent: 'episodic' for recalling specific past events, 'semantic' for factual questions about the user's profile, or 'general' for all other conversation."
    )
    

class AgentState(TypedDict):
    """
    Represents the state of the agent, including the input message,
    classification, reasoning behind the classification, and the final response.
    """
    message: str
    classification: Optional[str]
    reasoning: Optional[str]
    response: Optional[str]
    memory: Optional[str]
    scratchpad: Optional[str] 
    roleplay_count: int
    turn_count: int

    user_profile: Optional[UserProfile]
    semantic_memories: Optional[List[SemanticMemory]]
    episodic_memories: Optional[List[EpisodicMemory]]
    procedural_memories: Optional[List[ProceduralMemory]]
