from typing import TypedDict, Optional, List, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime


class AgentState(TypedDict, total=False):
    """State object for the agent graph."""
    message: str
    classification: Optional[str]
    reasoning: Optional[str]
    response: Optional[str]
    turn_count: Optional[int]
    roleplay_count: Optional[int]
    scratchpad: Optional[str]
    episodic_memories: Optional[List['EpisodicMemory']]
    semantic_memories: Optional[List['SemanticMemory']]
    procedural_memories: Optional[List['ProceduralMemory']]
    user_profile: Optional[dict]


class Router(BaseModel):
    """Router classification schema."""
    classification: Literal[
        "echo_respond",
        "roleplay", 
        "reflector",
        "philosopher",
        "complex_reasoning"
    ] = Field(..., description="The classification of the user's message")
    
    reasoning: str = Field(
        ..., 
        min_length=10,
        description="The reasoning behind the classification"
    )
    
    @validator('reasoning')
    def reasoning_must_be_detailed(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Reasoning must be at least 10 characters long')
        return v


class RetrievalClassifier(BaseModel):
    """Classification for memory retrieval strategy."""
    retrieval_type: Literal["episodic", "semantic", "general"] = Field(
        ..., 
        description="Type of memory retrieval needed"
    )


class UserProfile(BaseModel):
    """User profile schema."""
    name: str = Field(..., min_length=1, description="User's name")
    background: str = Field(..., min_length=10, description="User's background information")
    
    @validator('name', 'background')
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    communication_style: str = Field(..., min_length=5, description="User's communication preferences")
    emotional_baseline: str = Field(..., min_length=5, description="User's typical emotional state")
    
    @validator('communication_style', 'emotional_baseline')
    def validate_style_fields(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError('Field must be at least 5 characters long')
        return v.strip()


class SemanticMemory(BaseModel):
    """Semantic memory schema."""
    category: Literal[
        "preference",
        "trait", 
        "goal",
        "boundary",
        "summary"
    ] = Field(..., description="The type of information being stored.")

    content: str = Field(
        ..., 
        min_length=5,
        max_length=3000,  # Increased from 500 to 3000 characters to accommodate summaries
        description="The specific detail of the memory"
    )
    
    context: str = Field(
        ..., 
        min_length=10,
        max_length=1000,
        description="Additional context for the memory"
    )
    
    @validator('content', 'context')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    timestamp: str = Field(..., description="When this memory was created")


class EpisodicMemory(BaseModel):
    """Episodic memory schema."""
    user_message: str = Field(..., min_length=1, description="The user's message")
    ai_response: str = Field(..., min_length=1, description="The AI's response")
    
    @validator('user_message', 'ai_response')
    def validate_messages(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    timestamp: str = Field(..., description="When this interaction occurred")
    context: str = Field(default="", description="Additional context")


class ProceduralMemory(BaseModel):
    """Procedural memory schema."""
    trigger: str = Field(..., min_length=3, description="What triggers this procedure")
    action: str = Field(..., min_length=5, description="What action to take")
    context: str = Field(..., min_length=5, description="Context for this procedure")
    
    @validator('trigger', 'action', 'context')
    def validate_procedure_fields(cls, v):
        if not v or not v.strip():
            raise ValueError('Procedure field cannot be empty')
        return v.strip()
    
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Success rate of this procedure")
    timestamp: str = Field(..., description="When this procedure was learned")
    target_agent: str = Field(default="all", description="Which agent this applies to")