"""Unit tests for ``LLMClient`` against a mocked Messages API; no network, no key."""

import json
import logging
from collections.abc import Callable

import httpx2
import pytest
from anthropic import BadRequestError, DefaultAsyncHttpxClient
from pydantic import BaseModel

from review_river.llm import LLMClient, LLMOutputError

Handler = Callable[[httpx2.Request], httpx2.Response]


class Label(BaseModel):
    """Small schema to parse into, independent of the classifier's."""

    category: str
    confidence: float


def message_json(text: str, stop_reason: str = "end_turn") -> dict[str, object]:
    """Build a Messages API response body, as the server would send it.

    Parameters
    ----------
    text : str
        Text of the answer block, i.e. the JSON the model produced.
    stop_reason : str, default "end_turn"
        Why the model stopped, e.g. ``"max_tokens"`` or ``"refusal"``.

    Returns
    -------
    dict[str, object]
        Body with an empty thinking block before the text block, as Opus 5.5
        returns it, and fixed usage: 1,000 input and 200 output tokens.
    """
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def error_json(error_type: str) -> dict[str, object]:
    """Build a Messages API error body, e.g. ``"overloaded_error"``.

    Parameters
    ----------
    error_type : str
        API error type.

    Returns
    -------
    dict[str, object]
        Error body as the server sends it.
    """
    return {"type": "error", "error": {"type": error_type, "message": "mocked"}}


def make_llm(handler: Handler) -> LLMClient:
    """Build an ``LLMClient`` whose HTTP requests all go to ``handler``.

    Parameters
    ----------
    handler : Handler
        Called with each outgoing request; returns the mocked response.

    Returns
    -------
    LLMClient
        Client with a fake key and two retries; nothing reaches the network.
    """
    return LLMClient(
        api_key="sk-ant-test",
        model="claude-opus-5-5",
        effort="low",
        timeout_s=5.0,
        max_retries=2,
        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )


VALID_ANSWER = '{"category": "bug", "confidence": 0.9}'


class TestLLMClient:
    """Parsing, retries, output checks and logging of ``LLMClient.parse``."""

    async def test_returns_parsed_instance_and_sends_settings(self) -> None:
        """A valid answer comes back as ``Label``; system and effort reach the API."""
        calls: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            calls.append(request)
            return httpx2.Response(200, json=message_json(VALID_ANSWER))

        result = await make_llm(handler).parse(system="Classify.", prompt="Crashes", schema=Label)

        assert result == Label(category="bug", confidence=0.9)
        body = json.loads(calls[0].content)
        assert body["system"] == "Classify."
        assert body["output_config"]["effort"] == "low"
        assert body["output_config"]["format"]["type"] == "json_schema"

    async def test_retries_overloaded_then_succeeds(self) -> None:
        """A 529 is retried by the SDK; the second request succeeds."""
        calls: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            calls.append(request)
            if len(calls) == 1:
                # retry-after-ms keeps the SDK's backoff to 1 ms
                return httpx2.Response(
                    529, json=error_json("overloaded_error"), headers={"retry-after-ms": "1"}
                )
            return httpx2.Response(200, json=message_json(VALID_ANSWER))

        result = await make_llm(handler).parse(system="s", prompt="p", schema=Label)

        assert result.category == "bug"
        assert len(calls) == 2

    async def test_bad_request_is_not_retried(self) -> None:
        """A 400 propagates as ``BadRequestError`` after exactly one request."""
        calls: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            calls.append(request)
            return httpx2.Response(400, json=error_json("invalid_request_error"))

        with pytest.raises(BadRequestError):
            await make_llm(handler).parse(system="s", prompt="p", schema=Label)
        assert len(calls) == 1

    async def test_truncated_json_raises_output_error(self) -> None:
        """JSON cut off by ``max_tokens`` fails validation and maps to ``LLMOutputError``."""

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json=message_json('{"category": "bu', "max_tokens"))

        with pytest.raises(LLMOutputError):
            await make_llm(handler).parse(system="s", prompt="p", schema=Label)

    async def test_refusal_raises_output_error(self) -> None:
        """Valid JSON with ``stop_reason="refusal"`` is still rejected."""

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json=message_json(VALID_ANSWER, "refusal"))

        with pytest.raises(LLMOutputError, match="refusal"):
            await make_llm(handler).parse(system="s", prompt="p", schema=Label)

    async def test_no_text_block_raises_output_error(self) -> None:
        """A response with only a thinking block has no output to parse."""
        body = message_json("")
        body["content"] = [{"type": "thinking", "thinking": "", "signature": "sig"}]

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json=body)

        with pytest.raises(LLMOutputError, match="no text block"):
            await make_llm(handler).parse(system="s", prompt="p", schema=Label)

    async def test_logs_tokens_cost_and_request_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """One info line per call with tokens, cost and the request id.

        Cost: 1,000 input x $4 + 200 output x $20 per million tokens = $0.008.

        Parameters
        ----------
        caplog : pytest.LogCaptureFixture
            Captures log records emitted during the test.
        """
        caplog.set_level(logging.INFO, logger="review_river.llm")

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200, json=message_json(VALID_ANSWER), headers={"request-id": "req_123"}
            )

        await make_llm(handler).parse(system="s", prompt="p", schema=Label)

        assert "in=1000" in caplog.text
        assert "out=200" in caplog.text
        assert "cost_usd=0.008000" in caplog.text
        assert "request_id=req_123" in caplog.text
