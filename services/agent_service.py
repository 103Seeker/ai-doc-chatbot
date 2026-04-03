import os
from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from services.vector_service import search_chunks

# ── Groq LLM setup ──
# Get your free API key at: https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.2,
    max_tokens=1024,
)


# ─────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────

class AgentState(TypedDict):
    doc_id: str
    question: str
    retrieved_chunks: list[str]
    sources: list[str]
    raw_answer: str
    final_answer: str
    trace: list[str]


# ─────────────────────────────────────────
# AGENT 1 — Retriever Agent
# ─────────────────────────────────────────

def retriever_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", [])
    trace.append("🔍 Retriever Agent: Searching for relevant chunks...")

    results = search_chunks(state["doc_id"], state["question"], top_k=4)

    chunks = [r["text"] for r in results]
    sources = [f"Chunk {r['chunk_index']} (similarity: {r['score']})" for r in results]

    trace.append(f"✅ Retriever Agent: Found {len(chunks)} relevant chunks.")

    return {
        **state,
        "retrieved_chunks": chunks,
        "sources": sources,
        "trace": trace
    }


# ─────────────────────────────────────────
# AGENT 2 — Answer Agent
# ─────────────────────────────────────────

def answer_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", [])
    trace.append("🤖 Answer Agent: Generating answer using LLM...")

    context = "\n\n---\n\n".join(state["retrieved_chunks"])

    prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided document context.
If the answer is not in the context, say "I couldn't find this information in the document."

Document Context:
{context}

Question: {state["question"]}

Answer:"""

    response = llm.invoke(prompt)
    raw_answer = response.content.strip()

    trace.append("✅ Answer Agent: Answer generated.")

    return {
        **state,
        "raw_answer": raw_answer,
        "trace": trace
    }


# ─────────────────────────────────────────
# AGENT 3 — Summarizer Agent
# ─────────────────────────────────────────

def summarizer_agent(state: AgentState) -> AgentState:
    trace = state.get("trace", [])
    trace.append("✍️ Summarizer Agent: Polishing the final answer...")

    prompt = f"""You are an expert editor. Take the following answer and:
1. Make it clear and concise
2. Remove any repetition
3. Format it nicely with bullet points if needed
4. Keep all key information

Original Answer:
{state["raw_answer"]}

Polished Answer:"""

    response = llm.invoke(prompt)
    final_answer = response.content.strip()

    trace.append("✅ Summarizer Agent: Final answer ready.")

    return {
        **state,
        "final_answer": final_answer,
        "trace": trace
    }


# ─────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_agent)
    graph.add_node("answer", answer_agent)
    graph.add_node("summarizer", summarizer_agent)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


agent_graph = build_graph()


def run_agent(doc_id: str, question: str) -> dict:
    initial_state: AgentState = {
        "doc_id": doc_id,
        "question": question,
        "retrieved_chunks": [],
        "sources": [],
        "raw_answer": "",
        "final_answer": "",
        "trace": []
    }

    result = agent_graph.invoke(initial_state)

    return {
        "answer": result["final_answer"],
        "sources": result["sources"],
        "trace": result["trace"]
    }