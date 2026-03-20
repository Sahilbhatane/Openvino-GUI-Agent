"""Tests for screen analyzer utilities."""

import pytest

from agent.screen_analyzer import UIElement, build_element_map, build_elements_text


@pytest.fixture
def sample_elements():
    return [
        UIElement(id=1, control_type="Button", name="OK", bbox=(10, 20, 80, 50)),
        UIElement(id=2, control_type="Edit", name="Search", bbox=(100, 30, 300, 55)),
        UIElement(id=3, control_type="Hyperlink", name="Click here", bbox=(50, 100, 150, 120)),
    ]


class TestUIElement:
    def test_center(self):
        elem = UIElement(id=1, control_type="Button", name="OK", bbox=(10, 20, 80, 50))
        assert elem.center == (45, 35)

    def test_role_mapping(self):
        assert UIElement(id=1, control_type="Button", name="", bbox=(0, 0, 1, 1)).role == "button"
        assert UIElement(id=1, control_type="Edit", name="", bbox=(0, 0, 1, 1)).role == "input"
        assert UIElement(id=1, control_type="Hyperlink", name="", bbox=(0, 0, 1, 1)).role == "link"
        assert UIElement(id=1, control_type="MenuItem", name="", bbox=(0, 0, 1, 1)).role == "menuitem"

    def test_prompt_line(self):
        elem = UIElement(id=5, control_type="Button", name="Submit", bbox=(10, 20, 80, 50))
        line = elem.prompt_line()
        assert "[5]" in line
        assert "button" in line
        assert "Submit" in line

    def test_to_dict(self):
        elem = UIElement(id=1, control_type="Button", name="OK", bbox=(10, 20, 80, 50))
        d = elem.to_dict()
        assert d["id"] == 1
        assert d["role"] == "button"
        assert d["name"] == "OK"
        assert d["bbox"] == [10, 20, 80, 50]


class TestBuildElementMap:
    def test_builds_correct_map(self, sample_elements):
        m = build_element_map(sample_elements)
        assert m[1] == (45, 35)
        assert m[2] == (200, 42)
        assert m[3] == (100, 110)

    def test_empty_list(self):
        assert build_element_map([]) == {}


class TestBuildElementsText:
    def test_formats_elements(self, sample_elements):
        text = build_elements_text(sample_elements)
        assert "Elements on screen:" in text
        assert "[1]" in text
        assert "[2]" in text
        assert "Search" in text

    def test_empty_list(self):
        text = build_elements_text([])
        assert "No interactive elements" in text
