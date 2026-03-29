"""FastAPI surface (mocked VLM and controller — no model load)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    mock_instance = MagicMock()
    mock_instance.load = MagicMock()

    mock_controller = MagicMock()
    mock_controller.run.return_value = {
        "instruction": "open calculator",
        "status": "completed",
        "iterations": 2,
        "token_usage": {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_generation_time": 1.0,
            "avg_tokens_per_second": 5.0,
        },
        "history": [{"step": 1}],
    }

    with patch("api.server.VLMInference", return_value=mock_instance):
        with patch("api.server.AgentController", return_value=mock_controller):
            from api.server import app

            with TestClient(app) as client:
                yield client


def test_health_ready_after_startup(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "device" in body


def test_run_task_returns_controller_payload(api_client):
    r = api_client.post("/run-task", json={"instruction": "open calculator"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed"
    assert data["iterations"] == 2
    assert data["instruction"] == "open calculator"
