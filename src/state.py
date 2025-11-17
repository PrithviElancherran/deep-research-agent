from typing import List, TypedDict

from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    """
    This is the state that flows through the LangGraph.

    IMPORTANT: it has a `messages` field, as required by the guidelines.
    - `messages` will store:
        * the user's question (as a HumanMessage)
        * intermediate assistant/tool messages (like search results)
        * the final report (as an AIMessage)
    """

    messages: List[BaseMessage]
