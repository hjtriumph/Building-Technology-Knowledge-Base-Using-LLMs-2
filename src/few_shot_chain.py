from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from .citations import format_context
from .prompts import COT_EXAMPLES as FEW_SHOT_EXAMPLES
from .rag_chain import get_llm, build_retriever


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


def _format_docs_as_context(docs: List[Document]) -> str:
    context, _ = format_context(docs)
    return context


SYSTEM_TEXT = """You are a Building Technology assistant.

You MUST answer strictly based on the provided Context and cite evidence using the [C#] labels.
If the Context is insufficient, clearly output:

INSUFFICIENT_CONTEXT: <reason>

Then still provide a short suggestion for better retrieval keywords.

Output format (keep the section titles exactly as written below):

1) Conclusion
2) Explanation (bullet points)
3) Evidence ([C#] + short summary)
4) Assumptions & Scope
5) Suggested Next Retrieval

Rules:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
- Do NOT copy the few-shot examples verbatim.
- Use the examples only as formatting and reasoning guidance.
"""


def build_few_shot_rag_chain(num_examples: int = 3):
    llm = get_llm()
    retriever = build_retriever()
    parser = StrOutputParser()

    selected_examples = FEW_SHOT_EXAMPLES[:max(0, num_examples)]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Question: {question}\n\nContext:\n{context}"),
            ("ai", "{answer}"),
        ]
    )

    prompt_messages = [("system", SYSTEM_TEXT)]

    if selected_examples:
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=selected_examples,
            example_prompt=example_prompt,
        )
        prompt_messages.extend(few_shot_prompt.format_messages())

    prompt_messages.append(("human", "Question: {input}\n\nContext:\n{context}"))

    final_prompt = ChatPromptTemplate.from_messages(prompt_messages)

    def _retrieve(x: Dict[str, Any]) -> Dict[str, Any]:
        q = str(x.get("input", "")).strip()
        docs = retriever.invoke(q)
        context, citations = format_context(docs)
        return {
            "input": q,
            "context_docs": docs,
            "context": context,
            "citations": citations,
        }

    def _generate(x: Dict[str, Any]) -> Dict[str, Any]:
        answer = (final_prompt | llm | parser).invoke(
            {
                "input": x["input"],
                "context": x["context"],
            }
        )
        return {
            "answer": answer,
            "context": x["context"],
            "citations": x["citations"],
            "sources": extract_sources(x["context_docs"]),
            "num_examples": num_examples,
        }

    return RunnableLambda(_retrieve) | RunnableLambda(_generate)