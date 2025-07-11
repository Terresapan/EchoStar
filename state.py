from typing_extensions import TypedDict
from typing import Optional
from pydantic import BaseModel, Field
from typing import Literal

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
    ] = Field(
        description=(
            "The classification of the message:\n"
            "- 'echo_respond' for ambient or lightweight messages\n"
            "- 'roleplay' for flirtatious or metaphor-laced intimacy\n"
            "- 'reflector' for emotionally revealing or psychologically rich content\n"
            "- 'philosopher' for abstract, conceptual, or spiritual inquiries\n"
            "- 'complex_reasoning' for queries requiring multi-step thinking or memory synthesis\n"
        )
    )