import argparse
import os
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel

console = Console()

AVAILABLE_LANGUAGES = [
    "French",
    "Spanish",
    "German",
    "Italian",
    "Portuguese",
    "Japanese",
    "Arabic",
]


class DemoState(TypedDict, total=False):
    sentence: str
    coordinator_output: str
    final_answer: str


def build_model() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3.5-4B"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        temperature=0.1,
        max_completion_tokens=1200,
    )


@tool
def translate(text: str, target_language: str) -> str:
    """Translate the given text into one target language. Supported languages are fixed by the app."""
    if target_language not in AVAILABLE_LANGUAGES:
        return f"Unsupported language: {target_language}. Supported: {', '.join(AVAILABLE_LANGUAGES)}"

    model = build_model()
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    f"You are a translation tool. Translate the user's text into natural {target_language}. "
                    f"Return only the {target_language} translation."
                )
            ),
            HumanMessage(content=text),
        ]
    )
    return str(response.content).strip()


TOOLS = {translate.name: translate}


def coordinator_node(state: DemoState) -> DemoState:
    console.print("[bold cyan]Coordinator[/bold cyan] starting")
    tool_model = build_model().bind_tools([translate], tool_choice="required")
    planning_messages: list[Any] = [
        SystemMessage(
            content=(
                "You are a coordinator agent for translation. "
                f"You may use the translate tool with these hardcoded languages only: {', '.join(AVAILABLE_LANGUAGES)}. "
                "Decide which of those languages are useful for the user's request. "
                "If the user does not specify a target audience or language set, translate into all available languages. "
                "You must use the translate tool rather than translating from your own knowledge. "
                "In a single tool-using step, call translate once for each language you want included."
            )
        ),
        HumanMessage(content=state["sentence"]),
    ]

    response = tool_model.invoke(planning_messages)
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        fallback = "The coordinator did not invoke the translation tool."
        return {"coordinator_output": fallback, "final_answer": fallback}

    tool_messages: list[Any] = []
    for call in tool_calls:
        console.print(f"[cyan]Coordinator tool call[/cyan]: {call['name']} -> {call.get('args', {})}")
        result = TOOLS[call["name"]].invoke(call.get("args", {}))
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    formatter = build_model()
    final_response = formatter.invoke(
        [
            SystemMessage(
                content=(
                    "Format the completed translations. Return only this structure:\n"
                    "Original: <original sentence>\n"
                    "<Language>: <translation>\n"
                    "<Language>: <translation>"
                )
            ),
            HumanMessage(
                content=(
                    f"Original sentence: {state['sentence']}\n\n"
                    "Collected translations from tool calls:\n"
                    + "\n".join(msg.content for msg in tool_messages)
                )
            ),
        ]
    )

    final = str(final_response.content).strip()
    return {"coordinator_output": final, "final_answer": final}


def present_result_node(state: DemoState) -> DemoState:
    console.print("[bold green]Presenter[/bold green] starting")
    return {"final_answer": state.get("coordinator_output", "")}


def build_graph():
    graph = StateGraph(DemoState)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("present_result", present_result_node)
    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "present_result")
    graph.add_edge("present_result", END)
    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local translation agent demo.")
    parser.add_argument("sentence", nargs="?", default="The weather is beautiful today.")
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke({"sentence": args.sentence})

    console.print(Panel(", ".join(AVAILABLE_LANGUAGES), title="Available Languages"))
    console.print(Panel(str(result.get("coordinator_output", "")), title="Coordinator Output"))
    console.print(Panel(str(result.get("final_answer", "")), title="Final Output"))


if __name__ == "__main__":
    main()
