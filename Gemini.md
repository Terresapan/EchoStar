# EchoStar Agent with LangGraph and LangMem

This document outlines the architecture of the EchoStar personal companionship agent, built using LangGraph, LangMem, and an OpenAI model.

## 1. Project Goal

The goal is to create a multi-agent system where a central router classifies incoming messages and directs them to one of several specialized sub-agents for generating a response. The agent is designed to have a sophisticated, multi-layered memory system to enable long-term, meaningful conversations. For more complex queries, the system uses a planner/executor pattern.

## 2. Core Components

- **Model:** The application uses the `openai:gpt-4.1-mini` model for its language processing capabilities.
- **Router:** A LangChain chain that classifies messages into different categories to determine the appropriate sub-agent or workflow.
- **Sub-Agents/Nodes:** Several distinct nodes, each with a specific persona or function:
  - `echo_node`: For light, casual conversation.
  - `roleplay_node`: For romantic and roleplay scenarios.
  - `reflector_node`: For deep, emotional reflection.
  - `philosopher_node`: For abstract and intellectual discussions.
  - `planner_node`: For complex queries, this node outlines a multi-step plan.
  - `executor_node`: Executes the plan from the planner to generate a final response.
- **LangGraph:** Orchestrates the flow from the router to the appropriate sub-agent or planner/executor workflow.
- **LangMem:** Provides the long-term memory capabilities for the agent, including:
  - **Episodic Memory:** Stores the history of conversations.
  - **Semantic Memory:** Stores facts and user preferences.
  - **User Profile:** A comprehensive, evolving profile of the user.
- **Streamlit:** Used to create a web-based user interface for interacting with the agent.

## 3. File Structure

- `graph.py`: The main entry point for the application. It initializes the LLM, defines the graph, and runs the agent.
- `nodes.py`: Contains the definitions for all the nodes in the LangGraph, including the memory retrieval and saving nodes.
- `memory.py`: Initializes the `langmem` store and defines the memory managers for episodic, semantic, and profile memories.
- `schemas.py`: Defines the Pydantic models for the router's output, as well as the schemas for `EpisodicMemory`, `SemanticMemory`, and `UserProfile`.
- `state.py`: Defines the `AgentState` TypedDict for the LangGraph.
- `prompt.py`: Contains all the system and user prompts for the agent.
- `tool.py`: Defines the `mood_lift_tool` for simulated mental health interventions.
- `streamlit_app.py`: The Streamlit application for the user interface.
- `utils.py`: Contains utility functions for the Streamlit app, such as feedback saving and password checking.

## 4. Implementation Overview

### Agent State and Graph Definition

- The state for the graph is defined in `state.py`.
- The graph is defined and compiled in `graph.py`, which orchestrates the flow between nodes.

### Memory System

- The memory system is defined in `memory.py` and uses `langmem`.
- It has three types of memory: `EpisodicMemory`, `SemanticMemory`, and `UserProfile`.
- The `memory_retrieval_node` in `nodes.py` fetches relevant memories at the beginning of each turn.
- The `save_memories_node` in `nodes.py` saves memories at the end of each turn.

### Router Node

- The router node is defined in `nodes.py`.
- This node takes the `AgentState` as input, runs the classification using the OpenAI model, and updates the state with the `classification` result.

### Sub-Agent Nodes

- The sub-agent nodes (`echo_node`, `philosopher_node`, `reflector_node`, `roleplay_node`) are defined in `nodes.py`.
- Each node receives the `AgentState`, uses a specific system prompt from `prompt.py`, calls the LLM to generate a response, and updates the state.

### Planner/Executor Nodes

- For messages classified as `complex_reasoning`, the `planner_node` creates a step-by-step plan.
- The `executor_node` then executes this plan to generate a comprehensive response.

### Conditional Edges

- Conditional edges in `graph.py` use the `should_continue` function to route to the correct sub-agent node based on the `classification` in the `AgentState`.

### Main Application

- The `streamlit_app.py` file is the main entry point for the user. It runs the compiled LangGraph from `graph.py`.
- It takes an `incoming_message` and produces a final response from the selected sub-agent or workflow.
