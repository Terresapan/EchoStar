"""
Simple script to print the Mermaid diagram of the EchoStar workflow.
"""

import sys
import os
import streamlit as st


# Add the project root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set dummy environment variables for diagram generation (LLM won't actually be called)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-diagram-generation")

from src.agents.graph import create_graph
from src.agents.memory import get_memory_store
from langgraph.checkpoint.memory import MemorySaver

try:
    # Initialize the memory system
    print("🔄 Initializing memory system...")
    memory_system = get_memory_store()
    
    # Create the graph with memory system and checkpointer
    print("🔄 Creating workflow graph...")
    checkpointer = MemorySaver()
    app = create_graph(checkpointer, memory_system)
    
    # Print the Mermaid diagram
    print("🎯 EchoStar Workflow Diagram (Unified Reflector Architecture)")
    print("=" * 60)
    print(app.get_graph().draw_mermaid())
    print("=" * 60)
    print("✅ Key Changes:")
    print("• Merged 'reflector' and 'reflective_inquiry' into unified 'reflector'")
    print("• Reflector now adaptively handles both emotional support and pattern recognition")
    print("• Simplified classification - no more confusion between similar emotional categories")
    print("• Memory condensation triggers every 10 turns to summarize conversation history")
    
    # Optionally generate PNG if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--png":
        output_path = "mermaid.png"
        with open(output_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"📁 PNG diagram saved to {os.path.abspath(output_path)}")

except ImportError as e:
    print("❌ Error: Required dependencies not found.")
    print(f"Details: {e}")
    print("Please install missing dependencies:")
    print("  pip install pygraphviz")
    print("  # or for conda users:")
    print("  conda install pygraphviz")

except Exception as e:
    # Catching generic exceptions, often related to the Graphviz installation
    if "No such file or directory" in str(e) or "command not found" in str(e).lower():
        print("❌ Error: Graphviz not found.")
        print("Graphviz is a required system dependency for drawing graphs.")
        print("Please install it from: https://graphviz.org/download/")
        print("After installing Graphviz, also install pygraphviz:")
        print("  pip install pygraphviz")
    else:
        print(f"An unexpected error occurred: {e}")
        print("This might be due to missing Streamlit secrets or API keys.")
        print("Make sure you have a .streamlit/secrets.toml file with the required API keys.")