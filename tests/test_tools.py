"""Unit tests for each tool in isolation - no LLM, no agent loop involved."""

import pytest

from app.tools import calculator, weather, web_search


class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculator.calculate("2 + 2") == "4"

    def test_operator_precedence_and_parentheses(self):
        assert calculator.calculate("(12 + 8) * 3 / 4") == "15.0"

    def test_rejects_non_math_syntax(self):
        # This is the key safety test: nothing beyond numbers and arithmetic
        # operators should ever be allowed to run.
        with pytest.raises(ValueError):
            calculator.calculate("__import__('os').system('echo hi')")

    def test_rejects_garbage_input(self):
        with pytest.raises(ValueError):
            calculator.calculate("this is not math")


class TestWeather:
    def test_get_weather_happy_path(self, monkeypatch):
        # Fake the two HTTP calls get_weather makes, so the test needs no network.
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        def fake_get(url, params=None, timeout=None):
            if "geocoding" in url:
                return FakeResponse(
                    {"results": [{"name": "Dublin", "country": "Ireland", "latitude": 53.35, "longitude": -6.26}]}
                )
            return FakeResponse(
                {"current_weather": {"temperature": 14.2, "windspeed": 10.0, "weathercode": 2}}
            )

        monkeypatch.setattr(weather.requests, "get", fake_get)

        result = weather.get_weather("Dublin")
        assert "Dublin, Ireland" in result
        assert "14.2" in result
        assert "Partly cloudy" in result

    def test_get_weather_unknown_place(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": []}

        monkeypatch.setattr(weather.requests, "get", lambda *a, **k: FakeResponse())

        with pytest.raises(ValueError):
            weather.get_weather("Nowhereville")

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            weather.get_weather("   ")


class TestWebSearch:
    def test_search_web_formats_results(self, monkeypatch):
        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def text(self, query, max_results=3):
                return [{"title": "Example", "body": "Example snippet", "href": "https://example.com"}]

        monkeypatch.setattr(web_search, "DDGS", FakeDDGS)

        result = web_search.search_web("example query")
        assert "Example" in result
        assert "https://example.com" in result

    def test_rejects_empty_query(self):
        with pytest.raises(ValueError):
            web_search.search_web("   ")
