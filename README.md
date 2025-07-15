# EchoStar: An AI Personal Companionship Agent

EchoStar is a sophisticated, multi-agent personal companionship AI designed to engage in emotionally complex and psychologically nuanced conversations. Built with LangGraph, LangMem, and Streamlit, this agent uses a multi-layered memory system to foster long-term, meaningful interactions.

## 🚀 Features

- **Multi-Agent System**: A central router classifies incoming messages and directs them to one of several specialized sub-agents, each with a unique persona:
  - **Echo**: For light, casual conversation.
  - **Roleplay**: For romantic and imaginative scenarios.
  - **Reflector**: For deep, emotional reflection.
  - **Philosopher**: For abstract and intellectual discussions.
- **Sophisticated Memory**: The agent's memory is divided into four distinct layers:
  - **Episodic Memory**: Stores the history of conversations.
  - **Semantic Memory**: Stores facts, user preferences, and beliefs.
  - **User Profile**: A comprehensive, evolving profile of the user.
  - **Procedural Memory**: Learns from user feedback to adapt its behavior over time.
- **Planner/Executor Pattern**: For complex queries, the agent uses a planner to outline a multi-step plan and an executor to generate a comprehensive response.
- **Web-Based UI**: A user-friendly web interface built with Streamlit allows for easy interaction with the agent.

## 🛠️ Tech Stack

- **Backend**: Python, LangGraph, LangChain
- **Memory**: LangMem
- **Tracing**: LangSmith
- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4.1-mini

## 📦 Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/EchoStar.git
    cd EchoStar
    ```

2.  **Create a virtual environment and activate it:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys:**

    Create a `.streamlit/secrets.toml` file and add your OpenAI API key:

    ```toml
    [general]
    OPENAI_API_KEY = "your-openai-api-key"

    [tracing]
    LANGCHAIN_API_KEY = "your-langchain-api-key"
    ```

## 🏃‍♀️ Running the Application

To run the Streamlit application, use the following command:

```bash
streamlit run streamlit_app.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
