"""
Simple evals for the deep research agent.

These are not full-blown benchmarks, but they:
- Run the graph on several queries
- Check that a non-empty report is produced
- Check that the report includes a Sources section (grounding in web context)
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from src.config import load_config
from src.graph import create_graph, initial_state_from_query

TEST_QUERIES = [
    "What are the key challenges in scaling quantum computers?",
    "Summarize the current state of regulation around AI safety in the EU.",
    "Compare the environmental impact of electric cars vs. gasoline cars.",
]


def run_evals():
    load_dotenv()

    config = load_config()
    graph = create_graph(config)

    results = []

    for query in TEST_QUERIES:
        state = initial_state_from_query(query)
        final_state = graph.invoke(state)

        messages = final_state["messages"]
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]

        if not ai_messages:
            results.append((query, False, "No AI messages in final state"))
            continue

        final_report = ai_messages[-1].content

        # Very simple checks
        has_sources = "Sources" in final_report or "sources" in final_report
        length_ok = len(final_report.strip()) > 300  # at least some depth

        passed = has_sources and length_ok
        reason = []
        if not has_sources:
            reason.append("Missing 'Sources' section")
        if not length_ok:
            reason.append("Report too short")

        results.append((query, passed, "; ".join(reason) if reason else "OK"))

    # Print summary
    print("\n=== EVAL RESULTS ===")
    for query, passed, reason in results:
        status = "PASS" if passed else "FAIL"
        print(f"- [{status}] {query}")
        if reason:
            print(f"  Reason: {reason}")
    print("====================\n")


if __name__ == "__main__":
    run_evals()
