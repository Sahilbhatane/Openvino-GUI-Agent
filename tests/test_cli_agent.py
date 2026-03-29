"""CLI client: HTTP helper without a running server."""

from unittest.mock import MagicMock, patch

import cli_agent


def test_send_task_posts_json_and_returns_dict():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"status":"completed","iterations":1,"history":[]}'

    with patch("cli_agent.urllib.request.urlopen", return_value=mock_resp) as op:
        out = cli_agent.send_task("http://127.0.0.1:8000", "open calculator")

    assert out["status"] == "completed"
    op.assert_called_once()
    req = op.call_args[0][0]
    assert req.full_url.endswith("/run-task")


def test_main_prints_step_number_from_controller_history(capsys):
    """Controller history uses 'step'; CLI must display it correctly."""
    fake_result = {
        "status": "completed",
        "iterations": 1,
        "token_usage": {},
        "history": [
            {
                "step": 1,
                "thought": "done",
                "task_complete": True,
                "actions": [],
                "results": [],
            }
        ],
    }
    with patch.object(cli_agent, "send_task", return_value=fake_result):
        with patch.object(cli_agent.sys, "argv", ["cli_agent.py", "test"]):
            cli_agent.main()
    captured = capsys.readouterr().out
    assert "Iteration 1" in captured or "iteration 1" in captured.lower()
