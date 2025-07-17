"""
Enhanced memory nodes with comprehensive error context and monitoring.
Provides wrapper functions for existing memory operations with detailed tracking.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langgraph.store.memory import InMemoryStore

from .schemas import AgentState, SemanticMemory
from ..utils.memory_monitoring import create_memory_monitor, enhanced_memory_operation
from ..utils.error_context import ErrorSeverity, OperationType
from ..utils.logging_utils import get_logger
from config.manager import get_config_manager

logger = get_logger(__name__)


def enhanced_condense_memory_node(
    state: AgentState, 
    *, 
    llm: BaseChatModel, 
    store: Optional[InMemoryStore], 
    semantic_manager: Any, 
    episodic_manager: Any
) -> Dict[str, Any]:
    """
    Enhanced memory condensation with comprehensive error context and monitoring.
    Wraps the memory condensation process with detailed tracking and error handling.
    """
    
    # Extract context information
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id", "unknown")
    turn_count = state.get("turn_count", 0)
    
    # Create memory monitor with context
    monitor = create_memory_monitor(
        user_id=user_id,
        session_id=session_id,
        turn_count=turn_count,
        component="memory_condensation_node"
    )
    
    # Initialize comprehensive result tracking
    condensation_result = {
        "condensation_success": False,
        "semantic_storage": False,
        "episodic_storage": False,
        "cleanup_success": False,
        "turns_processed": 0,
        "error_message": None,
        "fallback_applied": False,
        "correlation_id": monitor.correlation_id,
        "performance_metrics": {},
        "error_context": []
    }
    
    with monitor.monitor_memory_condensation(
        user_id=user_id,
        session_id=session_id,
        turn_count=turn_count
    ) as condensation_ctx:
        
        try:
            # Step 1: Configuration retrieval with monitoring
            condensation_ctx['log_milestone']("configuration_retrieval_start")
            
            try:
                config_manager = get_config_manager()
                memory_config = config_manager.get_memory_config()
                TURNS_TO_SUMMARIZE = memory_config.turns_to_summarize
                
                condensation_ctx['add_metric']('turns_to_summarize', TURNS_TO_SUMMARIZE)
                condensation_ctx['log_milestone']("configuration_retrieved", turns_to_summarize=TURNS_TO_SUMMARIZE)
                
            except Exception as config_e:
                condensation_ctx['track_error'](config_e, ErrorSeverity.MEDIUM)
                TURNS_TO_SUMMARIZE = 10
                condensation_result["fallback_applied"] = True
                condensation_ctx['add_metric']('config_fallback_applied', True)
                
                logger.warning(
                    "Using fallback configuration for memory condensation",
                    correlation_id=monitor.correlation_id,
                    fallback_value=TURNS_TO_SUMMARIZE
                )
            
            # Step 2: Store availability check
            condensation_ctx['log_milestone']("store_availability_check")
            
            if store is None:
                error_msg = "Memory store unavailable"
                condensation_ctx['track_error'](
                    Exception(error_msg), 
                    ErrorSeverity.CRITICAL
                )
                condensation_result["error_message"] = error_msg
                condensation_result["fallback_applied"] = True
                condensation_ctx['add_metric']('store_unavailable', True)
                return condensation_result
            
            # Step 3: Memory retrieval with detailed monitoring
            condensation_ctx['log_milestone']("memory_retrieval_start")
            
            with monitor.monitor_memory_retrieval(
                namespace="('echo_star', 'Lily', 'collection')",
                limit=TURNS_TO_SUMMARIZE
            ) as retrieval_ctx:
                
                try:
                    recent_memories = store.search(
                        ('echo_star', 'Lily', 'collection'),
                        limit=TURNS_TO_SUMMARIZE
                    )
                    
                    memory_count = len(recent_memories) if recent_memories else 0
                    retrieval_ctx['add_metric']('memories_retrieved', memory_count)
                    condensation_ctx['add_metric']('memories_retrieved', memory_count)
                    
                    condensation_ctx['log_milestone'](
                        "memory_retrieval_completed", 
                        memories_found=memory_count
                    )
                    
                except Exception as search_e:
                    retrieval_ctx['track_error'](search_e)
                    condensation_ctx['track_error'](search_e, ErrorSeverity.HIGH)
                    
                    condensation_result["error_message"] = f"Memory retrieval failed: {str(search_e)}"
                    condensation_result["fallback_applied"] = True
                    condensation_ctx['add_metric']('retrieval_failed', True)
                    return condensation_result
            
            # Handle case where no memories are found
            if not recent_memories:
                condensation_ctx['log_milestone']("no_memories_found")
                condensation_result["fallback_applied"] = True
                condensation_ctx['add_metric']('no_memories_to_condense', True)
                return condensation_result
            
            # Step 4: Memory processing with error tracking
            condensation_ctx['log_milestone']("memory_processing_start", total_memories=len(recent_memories))
            
            dialogue_turns = []
            processing_errors = []
            
            for i, memory_item in enumerate(recent_memories):
                try:
                    value_dict = memory_item.value
                    content_dict = value_dict.get('content') if isinstance(value_dict, dict) else None
                    
                    if isinstance(content_dict, dict) and 'user_message' in content_dict and 'ai_response' in content_dict:
                        user_msg = content_dict['user_message']
                        ai_msg = content_dict['ai_response']
                        dialogue_turns.append(f"- User: {user_msg}, AI: {ai_msg}")
                    else:
                        logger.debug(f"Skipping non-episodic memory: item {i}")
                        
                except Exception as process_e:
                    processing_errors.append({
                        'item_index': i,
                        'error': str(process_e),
                        'error_type': type(process_e).__name__
                    })
                    condensation_ctx['track_error'](process_e, ErrorSeverity.LOW)
            
            # Log processing results
            condensation_ctx['add_metric']('dialogue_turns_extracted', len(dialogue_turns))
            condensation_ctx['add_metric']('processing_errors', len(processing_errors))
            
            if processing_errors:
                logger.warning(
                    "Memory processing encountered errors",
                    correlation_id=monitor.correlation_id,
                    error_count=len(processing_errors),
                    successful_items=len(dialogue_turns),
                    total_items=len(recent_memories)
                )
            
            if not dialogue_turns:
                condensation_ctx['log_milestone']("no_valid_dialogue_turns")
                condensation_result["error_message"] = "No valid dialogue turns found for condensation"
                condensation_result["fallback_applied"] = True
                condensation_ctx['add_metric']('no_valid_dialogue_turns', True)
                return condensation_result
            
            # Step 5: Summary generation with LLM monitoring
            condensation_ctx['log_milestone']("summary_generation_start", dialogue_turns=len(dialogue_turns))
            
            with monitor.monitor_llm_operation("memory_summary_generation") as llm_ctx:
                try:
                    from .prompts import consolidation_prompt
                    
                    formatted_memories = "\n".join(dialogue_turns)
                    llm_ctx['add_metric']('input_length', len(formatted_memories))
                    
                    summary_response = llm.invoke(consolidation_prompt.format(formatted_memories=formatted_memories))
                    summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
                    
                    if not summary or len(summary.strip()) == 0:
                        raise ValueError("Generated summary is empty")
                    
                    llm_ctx['add_metric']('output_length', len(summary))
                    condensation_ctx['add_metric']('summary_length', len(summary))
                    condensation_ctx['add_metric']('turns_processed', len(dialogue_turns))
                    
                    condensation_ctx['log_milestone'](
                        "summary_generation_completed",
                        summary_length=len(summary),
                        dialogue_turns_processed=len(dialogue_turns)
                    )
                    
                except Exception as summary_e:
                    llm_ctx['track_error'](summary_e)
                    condensation_ctx['track_error'](summary_e, ErrorSeverity.HIGH)
                    
                    condensation_result["error_message"] = f"Summary generation failed: {str(summary_e)}"
                    condensation_result["fallback_applied"] = True
                    condensation_ctx['add_metric']('summary_generation_failed', True)
                    return condensation_result
            
            # Step 6: Hybrid storage with detailed monitoring
            semantic_success = False
            episodic_success = False
            
            # Semantic memory storage
            condensation_ctx['log_milestone']("semantic_storage_start")
            
            with monitor.monitor_memory_storage(
                storage_type="semantic",
                namespace="('echo_star', 'Lily', 'facts')"
            ) as semantic_ctx:
                try:
                    current_timestamp = datetime.now().isoformat()
                    
                    semantic_memory = SemanticMemory(
                        category="summary",
                        content=summary,
                        context=f"Condensed conversation summary from {len(dialogue_turns)} recent turns",
                        importance=0.9,
                        timestamp=current_timestamp
                    )
                    
                    semantic_memory_key = str(uuid4())
                    semantic_memory_data = semantic_memory.model_dump()
                    
                    # Validate data structure
                    required_fields = ['category', 'content', 'context', 'importance', 'timestamp']
                    missing_fields = [field for field in required_fields if field not in semantic_memory_data]
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")
                    
                    # Store the data
                    store.put(
                        namespace=("echo_star", "Lily", "facts"),
                        key=semantic_memory_key,
                        value={
                            "content": semantic_memory_data,
                            "type": "SemanticMemory",
                            "category": "summary"
                        }
                    )
                    
                    # Verification
                    semantic_ctx['add_metric']('verification_attempted', True)
                    verification_result = store.search(
                        ("echo_star", "Lily", "facts"),
                        filter={"category": "summary"},
                        limit=1
                    )
                    
                    if verification_result:
                        semantic_ctx['add_metric']('verification_successful', True)
                        semantic_success = True
                        condensation_result["semantic_storage"] = True
                        
                        condensation_ctx['log_milestone'](
                            "semantic_storage_completed",
                            key=semantic_memory_key,
                            verification_successful=True
                        )
                    else:
                        logger.warning("Semantic storage verification failed")
                        semantic_ctx['add_metric']('verification_successful', False)
                        
                except Exception as semantic_e:
                    semantic_ctx['track_error'](semantic_e)
                    condensation_ctx['track_error'](semantic_e, ErrorSeverity.MEDIUM)
                    semantic_success = False
                    condensation_result["semantic_storage"] = False
            
            # Episodic memory storage
            condensation_ctx['log_milestone']("episodic_storage_start")
            
            with monitor.monitor_memory_storage(
                storage_type="episodic",
                namespace="('echo_star', 'Lily', 'collection')"
            ) as episodic_ctx:
                try:
                    current_timestamp = datetime.now().isoformat()
                    
                    episodic_messages = [
                        {"role": "user", "content": f"Condensed conversation summary from {len(dialogue_turns)} recent turns (timestamp: {current_timestamp})"},
                        {"role": "assistant", "content": f"Conversation Summary: {summary}. This condensed summary represents {len(dialogue_turns)} conversation turns."}
                    ]
                    
                    # Validate message structure
                    for i, msg in enumerate(episodic_messages):
                        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                            raise ValueError(f"Invalid message structure at index {i}")
                        if not msg['content'] or len(msg['content'].strip()) == 0:
                            raise ValueError(f"Empty content in message at index {i}")
                    
                    # Submit to episodic manager
                    if episodic_manager is None:
                        raise RuntimeError("Episodic manager is None")
                    
                    episodic_manager.submit({"messages": episodic_messages})
                    
                    # Verification
                    episodic_ctx['add_metric']('verification_attempted', True)
                    verification_memories = store.search(
                        ("echo_star", "Lily", "collection"),
                        limit=2
                    )
                    
                    # Check if our condensed summary is present
                    found_condensed = False
                    for mem in verification_memories:
                        if hasattr(mem, 'value') and isinstance(mem.value, dict):
                            content = mem.value.get('content', {})
                            if isinstance(content, dict):
                                ai_response = content.get('ai_response', '')
                                if 'Conversation Summary:' in ai_response and summary[:50] in ai_response:
                                    found_condensed = True
                                    break
                    
                    episodic_ctx['add_metric']('verification_successful', found_condensed)
                    episodic_success = True
                    condensation_result["episodic_storage"] = True
                    
                    condensation_ctx['log_milestone'](
                        "episodic_storage_completed",
                        verification_found=found_condensed
                    )
                    
                except Exception as episodic_e:
                    episodic_ctx['track_error'](episodic_e)
                    condensation_ctx['track_error'](episodic_e, ErrorSeverity.MEDIUM)
                    episodic_success = False
                    condensation_result["episodic_storage"] = False
            
            # Step 7: Cleanup with monitoring
            condensation_ctx['log_milestone']("cleanup_start", memories_to_cleanup=len(recent_memories))
            
            cleanup_success = False
            if semantic_success or episodic_success:
                try:
                    deleted_count = 0
                    failed_deletions = []
                    
                    for i, memory_item in enumerate(recent_memories):
                        try:
                            if hasattr(memory_item, 'key') and memory_item.key:
                                store.delete(("echo_star", "Lily", "collection"), memory_item.key)
                                deleted_count += 1
                            else:
                                failed_deletions.append({'index': i, 'error': 'Missing or empty key'})
                        except Exception as delete_e:
                            failed_deletions.append({
                                'index': i,
                                'key': getattr(memory_item, 'key', 'unknown'),
                                'error': str(delete_e)
                            })
                    
                    cleanup_success = deleted_count > 0
                    condensation_ctx['add_metric']('memories_deleted', deleted_count)
                    condensation_ctx['add_metric']('deletion_failures', len(failed_deletions))
                    
                    if failed_deletions:
                        logger.warning(
                            "Memory cleanup completed with failures",
                            correlation_id=monitor.correlation_id,
                            deleted_count=deleted_count,
                            failed_count=len(failed_deletions)
                        )
                    
                    condensation_ctx['log_milestone'](
                        "cleanup_completed",
                        deleted_count=deleted_count,
                        failed_count=len(failed_deletions)
                    )
                    
                except Exception as cleanup_e:
                    condensation_ctx['track_error'](cleanup_e, ErrorSeverity.LOW)
                    cleanup_success = False
            else:
                condensation_ctx['log_milestone']("cleanup_skipped", reason="no_successful_storage")
            
            condensation_result["cleanup_success"] = cleanup_success
            
            # Final status determination
            if semantic_success and episodic_success:
                condensation_result["condensation_success"] = True
                condensation_ctx['log_milestone']("condensation_fully_successful")
            elif semantic_success or episodic_success:
                condensation_result["condensation_success"] = True
                condensation_ctx['log_milestone']("condensation_partially_successful")
            else:
                condensation_result["condensation_success"] = False
                condensation_result["fallback_applied"] = True
                condensation_ctx['log_milestone']("condensation_failed")
            
            # Add performance metrics to result
            condensation_result["performance_metrics"] = monitor.get_operation_summary()
            
        except Exception as outer_e:
            condensation_ctx['track_error'](outer_e, ErrorSeverity.CRITICAL)
            condensation_result["error_message"] = f"Unexpected error: {str(outer_e)}"
            condensation_result["fallback_applied"] = True
            condensation_result["condensation_success"] = False
    
    # Final comprehensive logging
    logger.info(
        "Enhanced memory condensation completed",
        correlation_id=monitor.correlation_id,
        condensation_success=condensation_result["condensation_success"],
        semantic_storage=condensation_result["semantic_storage"],
        episodic_storage=condensation_result["episodic_storage"],
        cleanup_success=condensation_result["cleanup_success"],
        turns_processed=condensation_result["turns_processed"],
        fallback_applied=condensation_result["fallback_applied"],
        performance_summary=condensation_result["performance_metrics"]
    )
    
    return condensation_result