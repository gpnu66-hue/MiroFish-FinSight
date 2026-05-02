import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

from agents import (
    AgentState,
    data_fetcher,
    bull_analyst,
    bear_analyst,
    debater_node,
    revenue_visualizer,
    quality_inspector,
    reflect
)

load_dotenv()

def check_and_install_dependencies():
    try:
        import faiss
        print("[Setup] FAISS is already installed")
    except ImportError:
        print("[Setup] FAISS not found, installing...")
        os.system("pip install faiss-cpu")
        print("[Setup] FAISS installation completed")

check_and_install_dependencies()

def should_continue(state: AgentState) -> str:
    eval_score = state.get("eval_score", 0.0)
    iteration_count = state.get("iteration_count", 0)

    if eval_score >= 0.70:
        print(f"\n[Workflow] Strong score achieved: {eval_score:.4f}")
        return "end"
    elif iteration_count >= 1:  # 仅允许 1 次反思迭代
        print(f"\n[Workflow] Max reflection iterations reached. Score: {eval_score:.4f}")
        return "end"
    else:
        return "reflect"

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("DataFetcher", data_fetcher)
    workflow.add_node("BullAnalyst", bull_analyst)
    workflow.add_node("BearAnalyst", bear_analyst)
    workflow.add_node("Debater", debater_node)
    workflow.add_node("QualityInspector", quality_inspector)
    workflow.add_node("Reflect", reflect)
    workflow.add_node("RevenueVisualizer", revenue_visualizer)

    workflow.set_entry_point("DataFetcher")

    workflow.add_edge("DataFetcher", "BullAnalyst")
    workflow.add_edge("BullAnalyst", "BearAnalyst")
    workflow.add_edge("BearAnalyst", "Debater")
    workflow.add_edge("Debater", "QualityInspector")

    workflow.add_conditional_edges(
        "QualityInspector",
        should_continue,
        {
            "reflect": "Reflect",
            "end": "RevenueVisualizer"
        }
    )

    workflow.add_edge("RevenueVisualizer", END)
    workflow.add_edge("Reflect", "DataFetcher")

    return workflow.compile()

def run_financial_analysis(company_name: str):
    app = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=company_name)],
        "research_context": "",
        "eval_score": 0.0,
        "failure_reason": "",
        "iteration_count": 0,
        "suggested_queries": [],
        "bull_analysis": "",
        "bear_analysis": ""
    }

    print(f"Starting financial analysis for: {company_name}")
    print("=" * 80)

    try:
        final_state = app.invoke(initial_state, {"recursion_limit": 30})

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        print(f"\nFinal Evaluation Score: {final_state.get('eval_score', 0.0):.4f}")
        print(f"Status: {'PASSED' if final_state.get('eval_score', 0.0) >= 0.85 else 'NEEDS IMPROVEMENT'}")
        print(f"Total Iterations: {final_state.get('iteration_count', 0)}")
        print(f"Research Context Length: {len(final_state.get('research_context', ''))} characters")
        print(f"Multi-Agent Debate Mode: Enabled (Bull + Bear + Debater)")

        print("\n" + "-" * 80)
        print("MESSAGES:")
        print("-" * 80)
        for i, msg in enumerate(final_state["messages"], 1):
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"\n[{i}] {msg_type} Message:")
            print(msg.content)
            print("-" * 80)

        return final_state

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        company_name = " ".join(sys.argv[1:])
    else:
        company_name = "Apple Inc"

    print("MiroFish Financial Intelligence Engine")
    print("======================================")
    print(f"Target Company: {company_name}")
    print(f"Max Iterations: 3")
    print(f"Quality Threshold: 0.85")
    print()

    result = run_financial_analysis(company_name)

    if result:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed. Please check the error messages above.")
