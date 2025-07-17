#!/usr/bin/env python3
"""
Simple test script to verify InMemoryStore works independently
"""
import os
import streamlit as st
from langgraph.store.memory import InMemoryStore

# Import logging utilities
from logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

def test_store():
    """
    Test InMemoryStore functionality with comprehensive error handling and logging.
    
    Returns:
        InMemoryStore or None: The created store if successful, None if failed
    """
    logger.info("Starting InMemoryStore direct testing")
    
    # Set API key
    try:
        logger.debug("Attempting to load OpenAI API key from secrets")
        openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
        logger.info("OpenAI API key configured successfully", 
                   api_key_prefix=openai_api_key[:10] + "...")
    except KeyError as e:
        logger.error("Missing OpenAI API key in secrets", 
                    error=e,
                    missing_key=str(e))
        return None
    except Exception as e:
        logger.error("Unexpected error setting API key", 
                    error=e,
                    error_type=type(e).__name__)
        return None
    
    # Create store
    with logger.performance_timer("store_creation_test"):
        try:
            logger.debug("Creating InMemoryStore with embedding index")
            store = InMemoryStore(
                index={"embed": "openai:text-embedding-3-small"}
            )
            
            logger.info("InMemoryStore created successfully", 
                       store_type=type(store).__name__,
                       has_search_method=hasattr(store, 'search'))
            
            # Test search functionality
            logger.debug("Testing store search functionality")
            with logger.performance_timer("search_test"):
                result = store.search(("test", "namespace"))
                logger.info("Store search test completed", 
                           result_count=len(result),
                           result_type=type(result).__name__)
            
            logger.info("InMemoryStore test completed successfully")
            return store
            
        except Exception as e:
            logger.error("Failed to create or test InMemoryStore", 
                        error=e,
                        error_type=type(e).__name__)
            raise e

if __name__ == "__main__":
    test_store()