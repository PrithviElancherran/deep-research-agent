import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from .config import load_config
from .graph import create_graph, initial_state_from_query


def save_report_to_file(report_text: str, query: str) -> str:
    """
    Save the report to a Markdown file in the `reports/` directory.

    The filename includes a timestamp so each run is unique.
    """
    # Make sure the reports directory exists
    os.makedirs("reports", exist_ok=True)

    # Create a simple, safe filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Optional: shorten or clean query for the filename
    filename = f"reports/{timestamp}_report.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Deep Research Report\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write(report_text)

    return filename


def main():
    # Load environment variables from .env (OPENAI_API_KEY, TAVILY_API_KEY, etc.)
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python -m src.cli "your research question here"')
        sys.exit(1)

    # Combine all arguments as the query
    query = " ".join(sys.argv[1:])

    # Load configuration (model, search settings, etc.)
    config = load_config()

    # Create the graph
    graph = create_graph(config)

    # Build initial state with the user's query
    state = initial_state_from_query(query)

    # Run the graph. This returns the final state.
    final_state = graph.invoke(state)

    messages = final_state["messages"]

    # The final report should be the last AIMessage in messages
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    if not ai_messages:
        print("No AI response generated.")
        return

    final_report = ai_messages[-1].content

    # Print to terminal
    print("\n=== FINAL REPORT ===\n")
    print(final_report)
    print("\n====================\n")

    # Also save to file
    filepath = save_report_to_file(final_report, query)
    print(f"Report saved to: {filepath}")


if __name__ == "__main__":
    main()
