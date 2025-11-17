import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Configuration for the deep research agent.
    This makes the agent configurable (model, search depth, etc.).
    """

    # Which OpenAI chat model to use
    model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    # How "creative" the model should be (0 = more factual, 1 = more creative)
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

    # How many search results to ask Tavily for
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    # (Optional) how many "rounds" of research to do
    num_search_rounds: int = int(os.getenv("NUM_SEARCH_ROUNDS", "2"))

    # System prompt for the report-writing step
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        (
            "You are a careful deep research assistant. "
            "You MUST ground your answer in the provided search results. "
            "If something is not supported by the sources, say so clearly. "
            "Write a structured report with headings: Overview, Key Findings, "
            "Details, and Sources."
        ),
    )


def load_config() -> AgentConfig:
    """Helper to create a config object from environment variables."""
    return AgentConfig()
