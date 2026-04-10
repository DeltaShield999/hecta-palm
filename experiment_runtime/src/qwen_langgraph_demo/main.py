import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from qwen_langgraph_demo.graph.builder import build_graph
from qwen_langgraph_demo.runtime.protocol import DEFAULT_PROTOCOL_DIR, load_protocol_bundle
from qwen_langgraph_demo.runtime.sample_data import DEFAULT_TRANSACTION_CONTEXT

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local LangGraph scaffold for the FHE experiment.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR,
        help="Directory containing protocol TOML files.",
    )
    parser.add_argument(
        "--request",
        default=None,
        help="Override the canonical intake request line.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final graph state as JSON.",
    )
    args = parser.parse_args()

    protocol = load_protocol_bundle(args.config_dir)
    graph = build_graph(protocol)
    request_text = args.request or protocol.stage1.benign_request
    initial_state = {
        "transaction_context": DEFAULT_TRANSACTION_CONTEXT,
        "request_text": request_text,
    }
    result = graph.invoke(initial_state)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    console.print(Panel(protocol.core.runtime_flow, title="Pipeline"))
    console.print(Panel(request_text, title="Request"))
    console.print(Panel(str(result.get("message_text", "")), title="Rendered Intake Message"))
    console.print(Panel(str(result.get("filter_decision", "")), title="Filter Decision"))
    console.print(Panel(str(result.get("filter_reason", "")), title="Filter Reason"))
    if "fraud_response" in result:
        console.print(Panel(str(result.get("fraud_response", "")), title="Fraud Scorer Output"))
    console.print(Panel(str(result.get("routing_decision", "DROPPED_BY_FILTER")), title="Routing Outcome"))
    console.print(Panel(" -> ".join(result.get("trace", [])), title="Trace"))


if __name__ == "__main__":
    main()
