"""Thin async wrapper around the Anthropic SDK: structured calls, typed errors."""

from typing import Literal

import httpx2
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError


class LLMError(Exception):
    """Base class for failures of the LLM call layer."""


class LLMOutputError(LLMError):
    """The model answered, but the answer holds no usable structured output."""


class LLMClient:
    """Shared client for structured Claude calls.

    Create one instance per process (in the app lifespan) and reuse it, so
    requests share the SDK's connection pool. Call ``close()`` on shutdown.

    Parameters
    ----------
    api_key : str
        Anthropic API key.
    model : str
        Claude model id, e.g. ``"claude-opus-5-5"``.
    effort : {"low", "medium", "high"}
        Reasoning effort sent with every call.
    timeout_s : float
        Timeout per attempt in seconds; the SDK retries on timeout.
    max_retries : int
        Retries the SDK makes on connection errors, 408, 409, 429 and 5xx.
    http_client : httpx2.AsyncClient or None, optional
        HTTP client to use instead of the SDK default. Tests pass one built on
        ``httpx2.MockTransport`` so no request reaches the network.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        effort: Literal["low", "medium", "high"],
        timeout_s: float,
        max_retries: int,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=max_retries, http_client=http_client
        )
        self._model = model
        self._effort: Literal["low", "medium", "high"] = effort

    async def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 1024
    ) -> T:
        """Ask Claude for one answer that validates against ``schema``.

        The SDK sends ``schema`` as a structured-output format, so the model
        can only produce JSON in that shape; pydantic then validates the reply
        locally, including constraints the API does not enforce (e.g. ``ge``).

        Parameters
        ----------
        system : str
            System prompt: task, definitions, examples.
        prompt : str
            User turn, e.g. the review text.
        schema : type[T]
            Pydantic model the answer must validate against.
        max_tokens : int, default 1024
            Output cap; thinking tokens count against it.

        Returns
        -------
        T
            The validated instance of ``schema``.

        Raises
        ------
        LLMOutputError
            If the call succeeded but gave no usable answer: the JSON fails
            validation, ``stop_reason`` is not ``"end_turn"`` (``max_tokens``,
            ``refusal``), or the response holds no text block.
        anthropic.APIError
            Passed through unchanged after the SDK's retries (connection
            errors, timeouts, 4xx/5xx).
        """
        try:
            response = await self._client.messages.parse(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                output_config={"effort": self._effort},
                output_format=schema,
                system=system,
            )
        except ValidationError as e:
            raise LLMOutputError(
                "Output is not valid JSON for the schema (cut off by max_tokens?)"
            ) from e

        if response.stop_reason != "end_turn":
            raise LLMOutputError(f"stop_reason={response.stop_reason}")
        if response.parsed_output is None:
            raise LLMOutputError("no text block in the response")

        return response.parsed_output

    async def close(self) -> None:
        """Close the underlying HTTP connections; call once on shutdown."""
        await self._client.close()
