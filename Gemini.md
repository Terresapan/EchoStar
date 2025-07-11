# EchoStar Agent with Gemini and LangGraph

This document outlines the plan to build the EchoStar personal companionship agent using Google's Gemini model and LangGraph.

## 1. Project Goal

The goal is to create a multi-agent system where a central router classifies incoming messages and directs them to one of four specialized sub-agents for generating a response.

## 2. Core Components

- **Router:** A LangChain chain that uses a Gemini model to classify messages into one of four categories.
- **Sub-Agents:** Four distinct agents, each with a specific persona and purpose.
  - `EchoAgent`: For light, casual conversation.
  - `DarkHorsePrinceAgent`: For romantic and roleplay scenarios.
  - `InnerWitnessAgent`: For deep, emotional reflection.
  - `PhilosopherTwinAgent`: For abstract and intellectual discussions.
- **LangGraph:** To orchestrate the flow from the router to the appropriate sub-agent.

## 3. File Structure

- `graph.py`: The main entry point for the application. It initializes the LLM, defines the graph, and runs the agent.
- `state.py`: Defines the `AgentState` TypedDict for the LangGraph.
- `tool.py`: Defines the `Router` Pydantic model for the classification output.
- `nodes.py`: Contains the definitions for all the nodes in the LangGraph.
- `prompt.py`: Contains all the system and user prompts for the agent.

## 4. Implementation Steps

### Step 1: Switch to Gemini Model

- Update the LLM initialization in `graph.py` to use a Gemini model (e.g., `gemini-pro`).
- This will involve changing `init_chat_model("openai:gpt-4.1-mini")` to something like `init_chat_model("google:gemini-pro")`.
- Ensure all necessary environment variables (like `GOOGLE_API_KEY`) are set up in the `.env` file.

### Step 2: Define Agent States and Graph

- The state for the graph is defined in `state.py`.
- The graph is defined and compiled in `graph.py`.

### Step 3: Implement the Router Node

- The router node is defined in `nodes.py`.
- This node takes the `AgentState` as input, runs the classification, and updates the state with the `classification` result.

### Step 4: Implement the Sub-Agent Nodes

- The sub-agent nodes are defined in `nodes.py`.
- Each node will:
  - Receive the `AgentState`.
  - Use a specific system prompt tailored to its persona from `prompt.py`.
  - Call the Gemini LLM to generate a response.
  - Update the state with the generated response.

### Step 5: Define Conditional Edges

- The conditional edges are defined in `graph.py`.
- It uses the `should_continue` function to route to the correct sub-agent node based on the `classification` in the `AgentState`.

### Step 6: Compile the Graph

- The graph is compiled in `graph.py`.

### Step 7: Update `graph.py`

- The `graph.py` file is the main entry point and runs the compiled LangGraph.
- It will take the `incoming_message` and produce a final response from the selected sub-agent.
