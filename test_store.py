#!/usr/bin/env python3
"""
Simple test script to verify InMemoryStore works independently
"""
import os
import streamlit as st
from langgraph.store.memory import InMemoryStore

def test_store():
    print("=== TESTING INMEMORYSTORE DIRECTLY ===")
    
    # Set API key
    try:
        openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = openai_api_key
        print(f"API key set: {openai_api_key[:10]}...")
    except Exception as e:
        print(f"Error setting API key: {e}")
        return None
    
    # Create store
    try:
        store = InMemoryStore(
            index={"embed": "openai:text-embedding-3-small"}
        )
        print(f"Store created successfully: {store}")
        print(f"Store type: {type(store)}")
        print(f"Has search method: {hasattr(store, 'search')}")
        
        # Test search
        result = store.search(("test", "namespace"))
        print(f"Search test result: {result}")
        
        return store
        
    except Exception as e:
        print(f"Error creating store: {e}")
        print(f"Exception type: {type(e)}")
        raise e

if __name__ == "__main__":
    test_store()