from __future__ import annotations

from typing import TypedDict, List, Any, Dict
import json
import re

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .citations import format_context
from .pricing import estimate_cost
from .rag_chain import get_llm, build_retriever


class AgentState(TypedDict, total=False):
    question: str
    docs: List[Document]
    answer: str
    citations: List[Dict[str, Any]]
    context: str
    retrieved_chunk_ids: List[str]
    usage: Dict[str, int]

    # ReAct control fields
    thought: str
    action: str
    action_input: str
    observation: str

    # trace / memory
    scratchpad: str
    steps: List[Dict[str, str]]
    iterations: int


def _budget_exceeded(chunk_ids: List[str], limit: int) -> bool:
    return len(chunk_ids) >= int(limit)


def _usage_from(response: Any) -> Dict[str, int]:
    meta = getattr(response, "usage_metadata", None) or {}
    inp = int(meta.get("input_tokens", 0) or 0)
    out = int(meta.get("output_tokens", 0) or 0)
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": int(meta.get("total_tokens", inp + out) or (inp + out)),
    }


def _add_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {
        key: int(a.get(key, 0) or 0) + int(b.get(key, 0) or 0)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def _safe_json_loads(text: str) -> dict:
    """
    Robust JSON extraction:
    - strips ```json fences
    - extracts first {...} block if extra text exists
    - returns {} if parsing fails
    """
    if not text:
        return {}
    t = text.strip()

    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    try:
        return json.loads(t)
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    return {}


def _dedupe_docs_keep_order(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []

    for d in docs or []:
        md = d.metadata or {}
        key = (
            md.get("source", ""),
            md.get("page", None),
            md.get("chunk_id", None),
            (d.page_content or "").strip(),
        )
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _format_docs_as_context(docs: List[Document], max_context_chars: int = 12000) -> str:
    context, _ = format_context(docs)
    return context[:max_context_chars]


def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for d in docs or []:
        md = d.metadata or {}
        out.append(
            {
                "source": md.get("source", "unknown"),
                "page": md.get("page", None),
                "chunk_id": md.get("chunk_id", None),
            }
        )
    return out


REACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Building Technology agent that follows the ReAct paradigm.

Your job is to answer the user's question by deciding the next action yourself.

You can choose exactly one action each turn:
- "search": search the Building Technology knowledge base
- "finish": stop searching and write the final answer

Output JSON only with exactly these keys:
{{
  "thought": "...",
  "action": "search" or "finish",
  "action_input": "..."
}}

Rules:
- Use "search" if you still need evidence from the knowledge base.
- Use "finish" only when you already have enough evidence.
- Keep "thought" short and task-focused.
- Keep "action_input" short and useful.
- Do not output any text outside JSON.
- Always answer in the same language as the user's question.
""",
        ),
        (
            "human",
            """Question: {question}

Current scratchpad:
{scratchpad}

Decide the next step now.""",
        ),
    ]
)

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Building Technology assistant.

You MUST answer strictly based on the provided Context and cite evidence using the [C#] labels.
If the Context is insufficient, clearly output:

INSUFFICIENT_CONTEXT: <reason>

Then still provide a short suggestion for better retrieval keywords.

Output format (keep the section titles exactly as written below):

1) Conclusion
2) Explanation (bullet points)
3) Evidence ([C#] + short summary)
4) Assumptions & Limitations
5) Next Retrieval Suggestions

Rules:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
- Do NOT invent evidence not present in the Context.
- Be concise but specific.
""",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def build_agent_graph(
    max_steps: int = 4,
    max_context_chars: int = 12000,
    max_total_chunks: int = 18,
):
    """
    True ReAct-style Agentic RAG:

      think -> (search or finish)
         search -> observe -> think -> ...
         finish -> final_answer

    Here the LLM explicitly decides whether to search again or finish.
    """
    llm = get_llm()
    retriever = build_retriever()

    def _init_defaults(state: AgentState) -> AgentState:
        return {
            "question": state.get("question", ""),
            "docs": state.get("docs", []),
            "answer": state.get("answer", ""),
            "context": state.get("context", ""),
            "citations": state.get("citations", []),
            "retrieved_chunk_ids": state.get("retrieved_chunk_ids", []),
            "usage": state.get(
                "usage",
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
            "thought": state.get("thought", ""),
            "action": state.get("action", ""),
            "action_input": state.get("action_input", ""),
            "observation": state.get("observation", ""),
            "scratchpad": state.get("scratchpad", ""),
            "steps": state.get("steps", []),
            "iterations": int(state.get("iterations", 0)),
        }

    def think(state: AgentState) -> AgentState:
        state = _init_defaults(state)

        msg = REACT_PROMPT.format_messages(
            question=state["question"],
            scratchpad=state["scratchpad"] or "(empty)",
        )
        response = llm.invoke(msg)
        resp = response.content
        usage = _add_usage(state.get("usage", {}), _usage_from(response))
        data = _safe_json_loads(resp)

        thought = str(data.get("thought", "") or "").strip()
        action = str(data.get("action", "") or "").strip().lower()
        action_input = str(data.get("action_input", "") or "").strip()

        if action not in {"search", "finish"}:
            # fallback: prefer search if there is no evidence yet, otherwise finish
            action = "search" if not state.get("docs") else "finish"

        if action == "search" and not action_input:
            action_input = state["question"]

        if action == "finish" and not action_input:
            action_input = "Use the collected evidence to answer the question."

        new_steps = list(state.get("steps", []))
        new_steps.append(
            {
                "thought": thought,
                "action": action,
                "action_input": action_input,
            }
        )

        return {
            **state,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "steps": new_steps,
            "iterations": int(state.get("iterations", 0)) + 1,
            "usage": usage,
        }

    def search(state: AgentState) -> AgentState:
        state = _init_defaults(state)
        query = (state.get("action_input") or "").strip() or state["question"]

        new_docs = retriever.invoke(query)

        all_docs = _dedupe_docs_keep_order(list(state.get("docs", [])) + list(new_docs or []))
        if max_total_chunks and len(all_docs) > max_total_chunks:
            all_docs = all_docs[:max_total_chunks]
        chunk_ids = [
            str((d.metadata or {}).get("chunk_id", "")) for d in all_docs
        ]

        # build short observation text for the scratchpad
        obs_lines: List[str] = []
        for d in (new_docs or [])[:4]:
            md = d.metadata or {}
            src = md.get("source_basename") or md.get("source", "unknown")
            page = md.get("page", None)
            chunk_id = md.get("chunk_id", None)

            header = f"Source: {src}"
            if page is not None:
                header += f" | page={page}"
            if chunk_id is not None:
                header += f" | chunk={chunk_id}"

            snippet = (d.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            obs_lines.append(header + " | " + snippet)

        observation = "\n".join(obs_lines).strip()
        if not observation:
            observation = "No useful documents were retrieved."

        scratchpad = (state.get("scratchpad", "") or "").strip()
        new_block = (
            f"Thought: {state.get('thought', '')}\n"
            f"Action: search\n"
            f"Action Input: {query}\n"
            f"Observation: {observation}\n"
        )
        scratchpad = (scratchpad + "\n\n" + new_block).strip() if scratchpad else new_block

        new_steps = list(state.get("steps", []))
        if new_steps:
            new_steps[-1]["observation"] = observation

        return {
            **state,
            "docs": all_docs,
            "retrieved_chunk_ids": chunk_ids,
            "observation": observation,
            "scratchpad": scratchpad,
            "steps": new_steps,
        }

    def final_answer(state: AgentState) -> AgentState:
        state = _init_defaults(state)

        docs = _dedupe_docs_keep_order(state.get("docs", []))
        context, citations = format_context(docs)
        context = context[:max_context_chars]

        msg = FINAL_ANSWER_PROMPT.format_messages(
            question=state["question"],
            context=context,
        )
        response = llm.invoke(msg)
        ans = response.content
        usage = _add_usage(state.get("usage", {}), _usage_from(response))

        scratchpad = (state.get("scratchpad", "") or "").strip()
        finish_block = (
            f"Thought: {state.get('thought', '')}\n"
            f"Action: finish\n"
            f"Action Input: {state.get('action_input', '')}\n"
        )
        scratchpad = (scratchpad + "\n\n" + finish_block).strip() if scratchpad else finish_block

        return {
            **state,
            "answer": ans,
            "scratchpad": scratchpad,
            "docs": docs,
            "context": _format_docs_as_context(docs),
            "citations": citations,
            "usage": usage,
        }

    def route_after_think(state: AgentState):
        state = _init_defaults(state)

        # force finish when step or context budget is reached
        if int(state.get("iterations", 0)) >= max_steps:
            return "final_answer"
        if _budget_exceeded(state.get("retrieved_chunk_ids", []), max_total_chunks):
            return "final_answer"

        if state.get("action") == "search":
            return "search"

        return "final_answer"

    g = StateGraph(AgentState)
    g.add_node("think", think)
    g.add_node("search", search)
    g.add_node("final_answer", final_answer)

    g.set_entry_point("think")
    g.add_conditional_edges(
        "think",
        route_after_think,
        {
            "search": "search",
            "final_answer": "final_answer",
        },
    )
    g.add_edge("search", "think")
    g.add_edge("final_answer", END)

    return g.compile()