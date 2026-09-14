"""StateGraph wiring.

triage -> plan -> (Send fan-out, max_concurrency=3) retrieve_one -> grade
  -> [thin evidence & hop < MAX_HOPS: refine -> (fan-out) retrieve_one -> grade, repeat]
  -> xbrl_lookup -> extract -> compare -> synthesize -> verify
  -> [violations & synth_attempts < 2: synthesize, repeat]
  -> END
triage -> refuse -> END  (out of scope)

The grade -> refine -> retrieve cycle is why this is a graph and not a chain: retrieval
quality is checked and, if thin, actively repaired before synthesis ever runs on it.
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from filings_analyst.agent import nodes
from filings_analyst.agent.state import AgentState
from filings_analyst.obs.logging import get_logger
from filings_analyst.settings import get_settings

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("triage", nodes.triage)
    builder.add_node("make_plan", nodes.plan)
    builder.add_node("retrieve_one", nodes.retrieve_one)
    builder.add_node("grade", nodes.grade)
    builder.add_node("refine", nodes.refine)
    builder.add_node("xbrl_lookup", nodes.xbrl_lookup)
    builder.add_node("extract", nodes.extract)
    builder.add_node("compare", nodes.compare)
    builder.add_node("synthesize", nodes.synthesize)
    builder.add_node("verify", nodes.verify)
    builder.add_node("refuse", nodes.refuse)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage", nodes.route_after_triage, {"plan": "make_plan", "refuse": "refuse"}
    )
    # dispatch_retrieval returns list[Send], which the installed langgraph type stubs for
    # add_conditional_edges don't model precisely (they see Hashable | list[Hashable]) even
    # though Send is the documented way to fan out — cast rather than fight the stub.
    builder.add_conditional_edges(
        "make_plan", cast(Any, nodes.dispatch_retrieval), ["retrieve_one"]
    )
    builder.add_edge("retrieve_one", "grade")
    builder.add_conditional_edges(
        "grade", nodes.route_after_grade, {"refine": "refine", "xbrl_lookup": "xbrl_lookup"}
    )
    builder.add_conditional_edges("refine", cast(Any, nodes.dispatch_retrieval), ["retrieve_one"])
    builder.add_edge("xbrl_lookup", "extract")
    builder.add_edge("extract", "compare")
    builder.add_edge("compare", "synthesize")
    builder.add_edge("synthesize", "verify")
    builder.add_conditional_edges(
        "verify", nodes.route_after_verify, {"synthesize": "synthesize", "end": END}
    )
    builder.add_edge("refuse", END)

    return builder


def compile_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    return build_graph().compile(checkpointer=checkpointer)


@contextmanager
def sqlite_checkpointer() -> Iterator[SqliteSaver]:
    settings = get_settings()
    settings.ensure_data_dirs()
    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as saver:
        yield saver


def initial_state(question: str, thread_id: str) -> AgentState:
    return AgentState(
        question=question,
        thread_id=thread_id,
        scope=None,
        plan=[],
        hop=0,
        evidence=[],
        pending_retrieval_indices=[],
        xbrl_facts=[],
        facts=[],
        contradictions=[],
        draft=None,
        citations=[],
        violations=[],
        synth_attempts=0,
        errors=[],
    )


def ask(question: str, thread_id: str | None = None) -> dict[str, Any]:
    thread_id = thread_id or str(uuid.uuid4())
    with sqlite_checkpointer() as checkpointer:
        graph = compile_graph(checkpointer)
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "max_concurrency": 3,
        }
        logger.info("graph.ask", thread_id=thread_id)
        result: dict[str, Any] = graph.invoke(initial_state(question, thread_id), config=config)
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ask the filings analyst a question.")
    parser.add_argument("question")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument(
        "--mermaid", action="store_true", help="Print the graph as Mermaid and exit"
    )
    args = parser.parse_args()

    if args.mermaid:
        print(compile_graph().get_graph().draw_mermaid())
        return

    result = ask(args.question, thread_id=args.thread_id)

    print(result.get("draft", ""))
    print()
    print("Citations:")
    for c in result.get("citations", []):
        print(f"  {c.marker} -> {c.chunk_id} ({c.source_url})")
    if result.get("contradictions"):
        print("\nContradictions:")
        for c in result["contradictions"]:
            print(f"  [{c.kind}] {c.description}")
    if result.get("errors"):
        print("\nErrors/notes:")
        for e in result["errors"]:
            print(f"  - {e}")


if __name__ == "__main__":
    _cli()
