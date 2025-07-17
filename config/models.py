"""
Configuration models for EchoStar AI Simulator.
Defines Pydantic models for all configuration sections.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """Configuration for memory management system."""
    
    turns_to_summarize: int = Field(
        default=10, 
        ge=1, 
        le=100,
        description="Number of conversation turns before triggering memory condensation"
    )
    search_limit: int = Field(
        default=10, 
        ge=1, 
        le=50,
        description="Maximum number of memories to retrieve in searches"
    )
    enable_condensation: bool = Field(
        default=True,
        description="Whether to enable automatic memory condensation"
    )
    procedural_search_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of procedural memories to retrieve"
    )


class RoutingConfig(BaseModel):
    """Configuration for message routing and classification."""
    
    roleplay_threshold: int = Field(
        default=2, 
        ge=1, 
        le=10,
        description="Maximum number of consecutive roleplay sessions before intervention"
    )
    enable_fallback: bool = Field(
        default=True,
        description="Whether to enable fallback responses for unclassified messages"
    )
    classification_confidence_threshold: float = Field(
        default=0.8, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence threshold for message classification"
    )


class LLMConfig(BaseModel):
    """Configuration for Language Model settings."""
    
    model_name: str = Field(
        default="openai:gpt-4.1-mini",
        description="Name of the LLM model to use"
    )
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0,
        description="Temperature setting for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None, 
        ge=1,
        description="Maximum number of tokens in responses"
    )
    timeout: int = Field(
        default=30, 
        ge=1, 
        le=300,
        description="Timeout for LLM API calls in seconds"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging system."""
    
    level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    enable_file_logging: bool = Field(
        default=True,
        description="Whether to enable file-based logging"
    )
    log_file_path: str = Field(
        default="logs/echostar.log",
        description="Path to log file"
    )
    enable_performance_logging: bool = Field(
        default=False,
        description="Whether to enable performance timing logs"
    )


class EchoStarConfig(BaseModel):
    """Master configuration class for EchoStar AI Simulator."""
    
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"  # Prevent additional fields
        
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # Custom validation logic can be added here
        pass