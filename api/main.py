from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.config import settings
from src.pipelines import answer_agent, answer_llm, answer_rag

from .schemas import AskRequest

AnswerFn = Callable[[str, str, int | None], dict[str, Any]]
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def _default_answer(question: str, mode: str, k: int | None = None) -> dict[str, Any]:
    if mode == "llm":
        return answer_llm(question, structured=False)
    if mode == "llm_structured":
        return answer_llm(question, structured=True)
    if mode == "agent":
        return answer_agent(question)
    if mode in {"rag", "rag_0shot"}:
        return answer_rag(question, k=k)
    if mode.startswith("fewshot_no_cot_"):
        n = int(mode.rsplit("_", 1)[-1])
        return answer_rag(question, fewshot_n=n, no_cot=True)
    if mode.startswith("fewshot_"):
        n = int(mode.rsplit("_", 1)[-1])
        return answer_rag(question, fewshot_n=n)
    raise ValueError(f"unsupported mode: {mode}")


def create_app(answer_fn: AnswerFn | None = None) -> FastAPI:
    app = FastAPI(title="Building Technology KB")
    fn = answer_fn or _default_answer

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "model": settings.deepseek_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "collection_name": settings.collection_name,
            "search_type": settings.search_type,
            "k": settings.k,
            "fetch_k": settings.fetch_k,
        }

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        try:
            return fn(request.question, request.mode, request.k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
