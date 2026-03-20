"""Tests for the error categorization system."""

import pytest

from agent.errors import AgentError, ErrorCategory, StepErrors


class TestAgentError:
    def test_to_dict(self):
        err = AgentError(
            category=ErrorCategory.MODEL,
            message="Invalid JSON",
            recovery="Retrying with hint",
        )
        d = err.to_dict()
        assert d["category"] == "model"
        assert d["message"] == "Invalid JSON"
        assert d["recovery"] == "Retrying with hint"

    def test_to_dict_minimal(self):
        err = AgentError(category=ErrorCategory.EXECUTION, message="click failed")
        d = err.to_dict()
        assert "recovery" not in d
        assert "detail" not in d

    def test_str_format(self):
        err = AgentError(
            category=ErrorCategory.GROUNDING,
            message="Element not found",
            recovery="Using coordinate fallback",
        )
        s = str(err)
        assert "[GROUNDING]" in s
        assert "Element not found" in s
        assert "Recovery:" in s


class TestStepErrors:
    def test_add_and_has_errors(self):
        se = StepErrors()
        assert not se.has_errors
        se.add(ErrorCategory.OS, "Permission denied")
        assert se.has_errors
        assert len(se.errors) == 1

    def test_to_list(self):
        se = StepErrors()
        se.add(ErrorCategory.MODEL, "bad json", recovery="retry")
        se.add(ErrorCategory.EXECUTION, "click failed")
        result = se.to_list()
        assert len(result) == 2
        assert result[0]["category"] == "model"
        assert result[1]["category"] == "execution"

    def test_empty_to_list(self):
        se = StepErrors()
        assert se.to_list() == []
