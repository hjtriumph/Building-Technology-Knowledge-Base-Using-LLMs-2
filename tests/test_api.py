from fastapi.testclient import TestClient

from api.main import create_app


def test_health_and_config():
    client = TestClient(create_app())
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert "embedding_model" in cfg and "api_key" not in cfg


def test_ask_uses_injected_pipeline():
    def fake(question, mode, k=None):
        return {
            "answer": "ok [C1]",
            "citations": [{"label": "C1"}],
            "retrieved": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "cost_cny": 0.0},
            "latency_sec": 0.0,
            "mode": mode,
        }

    client = TestClient(create_app(answer_fn=fake))
    response = client.post("/api/ask", json={"question": "q", "mode": "rag_0shot"})
    assert response.status_code == 200
    assert response.json()["answer"] == "ok [C1]"
