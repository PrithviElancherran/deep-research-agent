from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from .config import AgentConfig
from .nodes import report_node, search_node
from .state import ResearchState


def create_graph(config: AgentConfig):
    """
    Create and compile the LangGraph for the deep research agent.

    Graph structure:
        entry -> search_node -> report_node -> END
    """
    builder = StateGraph(ResearchState)

    # Wrap node functions so they accept only `state`
    # (we close over `config` from outside)
    def search_node_with_config(state: ResearchState):
        return search_node(state, config)

    def report_node_with_config(state: ResearchState):
        return report_node(state, config)

    builder.add_node("search", search_node_with_config)
    builder.add_node("report", report_node_with_config)

    builder.set_entry_point("search")
    builder.add_edge("search", "report")
    builder.add_edge("report", END)

    graph = builder.compile()
    return graph


def initial_state_from_query(query: str) -> ResearchState:
    """
    Helper to build the starting state from a user query.

    Puts the user query into `messages` as a HumanMessage.
    """
    return ResearchState(
        messages=[
            HumanMessage(content=query),
        ]
    )
