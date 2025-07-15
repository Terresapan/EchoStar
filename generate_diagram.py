from graph import create_graph
from memory import get_memory_store
from langgraph.checkpoint.memory import MemorySaver
import os

try:
    # Initialize the memory system
    memory_system = get_memory_store()
    
    # Create the graph with memory system and checkpointer
    checkpointer = MemorySaver()
    app = create_graph(checkpointer, memory_system)
    
    # Generate the Mermaid PNG diagram from the compiled graph
    output_path = "mermaid.png"
    with open(output_path, "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    
    print(f"✅ Successfully generated workflow diagram and saved it to {os.path.abspath(output_path)}")

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