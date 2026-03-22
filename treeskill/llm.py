"""LLM Client — thin, multimodal-aware wrapper around ``openai.OpenAI``.

Handles serialization of ``Message`` objects (including image content parts)
into the format expected by the OpenAI Chat Completions API.

Features:
- Sync and async generation
- Automatic retry with exponential backoff for transient API errors
- Tool calling loop with max iteration guard
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import openai

from treeskill.builtin_tools import format_tool_result
from treeskill.config import GlobalConfig
from treeskill.schema import Message
from treeskill.tools import BaseTool

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 24

# Retry configuration
_MAX_RETRIES = 5
_BASE_DELAY = 1.0    # seconds
_MAX_DELAY = 60.0    # seconds

# HTTP status codes / error types that should trigger retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# 400 from load balancers (ALB) are transient, but true 400s from the API are not
_RETRYABLE_400_BODIES = {"alb", "bad gateway", "upstream"}


def _should_retry(exc: Exception) -> bool:
    """Determine if an API error is retryable."""
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code in _RETRYABLE_STATUS_CODES:
            return True
        # Some 400s from load balancers (ALB/nginx) are transient
        if exc.status_code == 400:
            body = str(exc).lower()
            return any(kw in body for kw in _RETRYABLE_400_BODIES)
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return False


def _get_retry_after(exc: Exception) -> Optional[float]:
    """Extract retry-after hint from API error headers, if available."""
    if isinstance(exc, openai.APIStatusError):
        headers = getattr(exc, "response", None)
        if headers is not None:
            retry_after = getattr(headers, "headers", {}).get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
    return None


class LLMClient:
    """Stateless wrapper around the OpenAI Python SDK with retry support.

    Parameters
    ----------
    config : GlobalConfig
        Framework configuration (API key, base URL, model names, etc.).
    """

    def __init__(self, config: GlobalConfig) -> None:
        self._config = config
        self._client = openai.OpenAI(
            api_key=config.llm.api_key.get_secret_value(),
            base_url=config.llm.base_url,
        )
        self._async_client = openai.AsyncOpenAI(
            api_key=config.llm.api_key.get_secret_value(),
            base_url=config.llm.base_url,
        )

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    def _call_with_retry(self, fn: Callable, **kwargs) -> Any:
        """Call *fn* with retry on transient API errors.

        Uses exponential backoff with jitter. Respects ``retry-after``
        headers when present.
        """
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn(**kwargs)
            except Exception as exc:
                last_exc = exc
                if not _should_retry(exc):
                    raise

                # Calculate delay
                retry_after = _get_retry_after(exc)
                if retry_after:
                    delay = retry_after
                else:
                    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                    delay *= 0.5 + random.random()  # jitter

                logger.warning(
                    "API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    async def _acall_with_retry(self, fn: Callable, **kwargs) -> Any:
        """Async version of retry wrapper."""
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await fn(**kwargs)
            except Exception as exc:
                last_exc = exc
                if not _should_retry(exc):
                    raise

                retry_after = _get_retry_after(exc)
                if retry_after:
                    delay = retry_after
                else:
                    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                    delay *= 0.5 + random.random()

                logger.warning(
                    "API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, exc, delay,
                )
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        tools: Optional[Mapping[str, BaseTool]] = None,
        on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> Message:
        """Send a chat completion request and return the response as a ``Message``.

        Automatically retries on transient API errors (429, 5xx) with
        exponential backoff.
        """
        model = model or self._config.llm.model
        api_messages = [msg.to_api_dict() for msg in messages]
        tool_defs = None
        if tools:
            tool_defs = [
                {"type": "function", "function": tool.to_schema()}
                for tool in tools.values()
            ]
        temperature = kwargs.pop("temperature", self._config.llm.temperature)

        for _ in range(_MAX_TOOL_ITERATIONS):
            request_kwargs: Dict[str, Any] = dict(kwargs)
            request_kwargs.update({
                "model": model,
                "messages": api_messages,
                "temperature": temperature,
            })
            if self._config.llm.extra_body:
                request_kwargs.setdefault("extra_body", {})
                request_kwargs["extra_body"].update(self._config.llm.extra_body)
            if tool_defs:
                request_kwargs["tools"] = tool_defs
                request_kwargs["tool_choice"] = "auto"

            completion = self._call_with_retry(
                self._client.chat.completions.create,
                **request_kwargs,
            )

            choice = completion.choices[0]
            assistant_message = choice.message
            tool_calls = getattr(assistant_message, "tool_calls", None) or []

            if not tool_calls:
                return Message(
                    role="assistant",
                    content=assistant_message.content or "",
                )

            api_messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    call.model_dump() if hasattr(call, "model_dump") else {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            })

            for call in tool_calls:
                tool_name = call.function.name
                if on_tool_event:
                    on_tool_event("start", {"name": tool_name, "arguments": call.function.arguments})
                try:
                    parsed_args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = {"error": f"Invalid tool arguments: {exc}"}
                else:
                    tool = tools.get(tool_name) if tools else None
                    if tool is None:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    else:
                        try:
                            result = tool.execute(**parsed_args)
                        except Exception as exc:
                            result = {"error": str(exc)}

                if on_tool_event:
                    on_tool_event("finish", {"name": tool_name, "result": format_tool_result(result)})

                api_messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": format_tool_result(result),
                })

        raise RuntimeError(
            f"Tool-calling loop exceeded the maximum number of iterations ({_MAX_TOOL_ITERATIONS})"
        )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def agenerate(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Message:
        """Async version of ``generate()`` (no tool support).

        Used for parallel candidate generation and scoring in APO.
        """
        model = model or self._config.llm.model
        api_messages = [msg.to_api_dict() for msg in messages]
        temperature = kwargs.pop("temperature", self._config.llm.temperature)

        request_kwargs: Dict[str, Any] = dict(kwargs)
        request_kwargs.update({
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
        })
        if self._config.llm.extra_body:
            request_kwargs.setdefault("extra_body", {})
            request_kwargs["extra_body"].update(self._config.llm.extra_body)

        completion = await self._acall_with_retry(
            self._async_client.chat.completions.create,
            **request_kwargs,
        )

        choice = completion.choices[0]
        return Message(
            role="assistant",
            content=choice.message.content or "",
        )

    def generate_batch(
        self,
        message_batches: List[List[Message]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Message]:
        """Generate responses for multiple message lists in parallel.

        Uses ``asyncio.gather`` internally. Safe to call from sync code.
        """
        async def _run():
            tasks = [
                self.agenerate(msgs, model=model, **kwargs)
                for msgs in message_batches
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                results = pool.submit(lambda: asyncio.run(_run())).result()
        else:
            results = asyncio.run(_run())

        # Convert exceptions to error messages
        messages: List[Message] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Parallel generation failed: %s", r)
                messages.append(Message(role="assistant", content=""))
            else:
                messages.append(r)
        return messages
