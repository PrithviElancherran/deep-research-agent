# Deep Research Agent (LangGraph + LangChain)

A simple, configurable **deep research agent** built with **LangGraph** and **LangChain**.

- Takes a **user query**
- Performs **web search (Tavily)** to gather real-world context
- Returns a **structured report** grounded in those search results (with citations)

---

## Features

- ✅ Uses **LangGraph** with a `messages` field in state
- ✅ Accepts a **query** and returns a **final report** as an assistant message
- ✅ Uses **Tavily** web search via the modern `langchain-tavily` integration
- ✅ Report is **grounded in web context** and includes a **Sources** section
- ✅ Agent is **configurable** (model, temperature, number of searches, etc.)
- ✅ Includes **simple evals** to check basic report quality

---

## Tech Stack

- Python 3.11+
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI via `langchain-openai`](https://python.langchain.com/docs/integrations/llms/openai)
- [Tavily Search via `langchain-tavily`](https://github.com/tavily-ai/langchain-tavily)
- `python-dotenv` for loading environment variables

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/PrithviElancherran/deep-research-agent.git
cd deep-research-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\Activate   # Windows (PowerShell)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root with the following content:

```env
# Required API keys
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

# Optional configuration
OPENAI_MODEL=gpt-4.1-mini                 # OpenAI model to use
OPENAI_TEMPERATURE=0.2                    # Response creativity (0 = more factual)

MAX_SEARCH_RESULTS=5                      # Number of web results per Tavily search
NUM_SEARCH_ROUNDS=2                       # Number of search rounds (depth)

SYSTEM_PROMPT=You are a careful deep research assistant. You MUST ground your answer in the provided search results. If something is not supported by the sources, say so clearly. Write a structured report with headings: Overview, Key Findings, Details, and Sources.
```

> **Note:**  
> `.env` is included in `.gitignore`, so your API keys will **not** be committed to GitHub.

---

## Running the Agent (CLI)

Run the agent:

```bash
python -m src.cli "What are the key challenges in scaling quantum computers?"
```

---

## Configuration

| Variable | Description |
|---------|-------------|
| `OPENAI_MODEL` | Chat model (e.g., gpt-4.1-mini) |
| `OPENAI_TEMPERATURE` | Creativity level |
| `MAX_SEARCH_RESULTS` | Max results per search |
| `NUM_SEARCH_ROUNDS` | Number of search rounds |
| `SYSTEM_PROMPT` | Instructions for writing the report |

---

## Architecture

### State

```python
class ResearchState(TypedDict):
    messages: List[BaseMessage]
```

### Nodes

- **search_node** → Tavily search  
- **report_node** → OpenAI structured report  

### Graph Flow

```
entry → search → report → END
```

---

## Evals

Run:

```bash
python -m tests.test_evals
```

---

## Future Improvements

- Add planning and critique nodes  
- Add deeper evals  
- Add UI  

---
