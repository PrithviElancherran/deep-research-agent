from typing import Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from .config import AgentConfig
from .state import ResearchState


def make_search_tool(max_results: int) -> TavilySearch:
    """
    Create a Tavily search tool using the new `langchain-tavily` package.
    """
    return TavilySearch(
        max_results=max_results,
        topic="general",
        include_answer=True,  # ask Tavily to give a short answer too
        include_raw_content=False,  # keep responses compact
    )


# def search_node(state: ResearchState, config: AgentConfig) -> Dict:
#     """
#     Node: perform web search using Tavily based on the latest user query.

#     - Reads the last HumanMessage from `state["messages"]`
#     - Calls Tavily with that text
#     - Appends an AIMessage that summarizes the search results
#     """
#     messages = state["messages"]

#     # Find the last user (human) message
#     user_messages = [m for m in messages if isinstance(m, HumanMessage)]
#     if not user_messages:
#         # No user message — nothing to search for
#         return {"messages": messages}

#     latest_user_message = user_messages[-1]
#     query = latest_user_message.content

#     # Create the Tavily search tool
#     search_tool = make_search_tool(config.max_search_results)

#     # Call Tavily. The tool returns a dict with an "answer" and a "results" list
#     search_response = search_tool.invoke({"query": query})

#     answer = search_response.get("answer")
#     results = search_response.get("results", [])

#     lines = ["SEARCH_RESULTS:"]

#     if answer:
#         lines.append("SUMMARY_ANSWER:")
#         lines.append(answer)
#         lines.append("")

#     for i, result in enumerate(results, start=1):
#         title = result.get("title") or "No title"
#         url = result.get("url") or "No URL"
#         snippet = result.get("content") or ""

#         lines.append(f"{i}. {title}")
#         lines.append(f"   URL: {url}")
#         if snippet:
#             lines.append(f"   Snippet: {snippet[:300]}")  # limit length
#         lines.append("")

#     summary_text = "\n".join(lines)

#     # Add an assistant message that includes the search results
#     search_message = AIMessage(
#         content=summary_text,
#         name="web_search",
#     )

#     new_messages = messages + [search_message]
#     return {"messages": new_messages}


def search_node(state: ResearchState, config: AgentConfig) -> Dict:
    """
    Node: perform web search using Tavily based on the latest user query.

    - Reads the last HumanMessage from `state["messages"]`
    - Calls Tavily one or more times (config.num_search_rounds)
    - Appends an AIMessage that summarizes all search results
    """
    messages = state["messages"]

    # Find the last user (human) message
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not user_messages:
        # No user message — nothing to search for
        return {"messages": messages}

    latest_user_message = user_messages[-1]
    query = latest_user_message.content

    # Create the Tavily search tool
    search_tool = make_search_tool(config.max_search_results)

    # Collect results across multiple rounds
    all_results = []
    all_answers = []

    for round_idx in range(config.num_search_rounds):
        search_response = search_tool.invoke({"query": query})

        answer = search_response.get("answer")
        results = search_response.get("results", [])

        if answer:
            all_answers.append(f"Round {round_idx + 1}: {answer}")

        all_results.extend(results)

    # Deduplicate results by URL (simple but effective)
    seen_urls = set()
    deduped_results = []
    for r in all_results:
        url = r.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped_results.append(r)

    # Build a readable summary string for the model
    lines = ["SEARCH_RESULTS:"]

    if all_answers:
        lines.append("SUMMARY_ANSWERS_FROM_SEARCH:")
        lines.extend(all_answers)
        lines.append("")

    for i, result in enumerate(deduped_results, start=1):
        title = result.get("title") or "No title"
        url = result.get("url") or "No URL"
        snippet = result.get("content") or ""

        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   Snippet: {snippet[:300]}")  # limit length
        lines.append("")

    summary_text = "\n".join(lines)

    # Add an assistant message that includes the search results
    search_message = AIMessage(
        content=summary_text,
        name="web_search",
    )

    new_messages = messages + [search_message]
    return {"messages": new_messages}


def report_node(state: ResearchState, config: AgentConfig) -> Dict:
    """
    Node: use OpenAI to write a structured report based on the messages,
    which now include SEARCH_RESULTS from the previous node.

    - Adds a SystemMessage with instructions
    - Feeds all messages (user + search results) to the model
    - Appends the final report as an AIMessage
    """
    messages = state["messages"]

    # Initialize the OpenAI chat model
    llm = ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
    )

    system_msg = SystemMessage(content=config.system_prompt)

    # The model will see:
    # - Our system instructions
    # - The conversation with the user
    # - The SEARCH_RESULTS assistant message from the search_node
    model_input = [system_msg] + messages

    report: AIMessage = llm.invoke(model_input)

    new_messages = messages + [report]
    return {"messages": new_messages}
