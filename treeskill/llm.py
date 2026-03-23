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
    """Multi-protocol wrapper supporting OpenAI and Anthropic APIs.

    Supports separate API endpoints for actor / judge / rewrite roles.
    Clients are lazily created and cached by (protocol, base_url, api_key).

    Parameters
    ----------
    config : GlobalConfig
        Framework configuration (API key, base URL, model names, etc.).
    """

    def __init__(self, config: GlobalConfig) -> None:
        self._config = config
        # Cache clients by (protocol, base_url, api_key) to avoid duplicates
        self._clients: Dict[tuple, Any] = {}
        self._async_clients: Dict[tuple, Any] = {}

    def _resolve_endpoint(self, role: Optional[str] = None):
        """Resolve (api_key, base_url, model, temperature, extra_body, protocol).

        Roles: "actor" (default), "judge", "rewrite".
        Falls back: rewrite → judge → actor.
        """
        llm = self._config.llm

        if role == "judge":
            api_key = (llm.judge_api_key or llm.api_key).get_secret_value()
            base_url = llm.judge_base_url or llm.base_url
            model = llm.judge_model
            temperature = llm.judge_temperature if llm.judge_temperature is not None else llm.temperature
            extra_body = llm.judge_extra_body if llm.judge_extra_body is not None else llm.extra_body
            protocol = llm.judge_protocol or llm.protocol
        elif role == "rewrite":
            api_key = (llm.rewrite_api_key or llm.judge_api_key or llm.api_key).get_secret_value()
            base_url = llm.rewrite_base_url or llm.judge_base_url or llm.base_url
            model = llm.rewrite_model or llm.judge_model
            temperature = (
                llm.rewrite_temperature if llm.rewrite_temperature is not None
                else llm.judge_temperature if llm.judge_temperature is not None
                else llm.temperature
            )
            extra_body = (
                llm.rewrite_extra_body if llm.rewrite_extra_body is not None
                else llm.judge_extra_body if llm.judge_extra_body is not None
                else llm.extra_body
            )
            protocol = llm.rewrite_protocol or llm.judge_protocol or llm.protocol
        else:  # actor / default
            api_key = llm.api_key.get_secret_value()
            base_url = llm.base_url
            model = llm.model
            temperature = llm.temperature
            extra_body = llm.extra_body
            protocol = llm.protocol

        return api_key, base_url, model, temperature, extra_body, protocol

    def _get_client(self, role: Optional[str] = None) -> Any:
        """Get or create a sync client for the given role."""
        api_key, base_url, _, _, _, protocol = self._resolve_endpoint(role)
        key = (protocol, base_url, api_key)
        if key not in self._clients:
            if protocol == "anthropic":
                import anthropic
                self._clients[key] = anthropic.Anthropic(
                    api_key=api_key, base_url=base_url,
                )
            else:
                self._clients[key] = openai.OpenAI(api_key=api_key, base_url=base_url)
        return self._clients[key]

    def _get_async_client(self, role: Optional[str] = None) -> Any:
        """Get or create an async client for the given role."""
        api_key, base_url, _, _, _, protocol = self._resolve_endpoint(role)
        key = (protocol, base_url, api_key)
        if key not in self._async_clients:
            if protocol == "anthropic":
                import anthropic
                self._async_clients[key] = anthropic.AsyncAnthropic(
                    api_key=api_key, base_url=base_url,
                )
            else:
                self._async_clients[key] = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._async_clients[key]

    # ------------------------------------------------------------------
    # Anthropic protocol adapters
    # ------------------------------------------------------------------

    def _anthropic_generate(
        self, client: Any, model: str, messages: List[Message],
        temperature: float, extra_body: Optional[Dict] = None,
        max_tokens: int = 8192,
    ) -> Message:
        """Call Anthropic Messages API and return a Message."""
        api_messages = [msg.to_api_dict() for msg in messages]
        # Extract system message
        system_text = ""
        filtered = []
        for m in api_messages:
            if m["role"] == "system":
                system_text = m["content"] if isinstance(m["content"], str) else str(m["content"])
            else:
                filtered.append(m)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": filtered,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text

        resp = self._call_with_retry(client.messages.create, **kwargs)

        # Extract text from content blocks (skip thinking blocks)
        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        return Message(role="assistant", content="".join(text_parts).strip())

    async def _anthropic_agenerate(
        self, client: Any, model: str, messages: List[Message],
        temperature: float, extra_body: Optional[Dict] = None,
        max_tokens: int = 8192,
    ) -> Message:
        """Async Anthropic Messages API call."""
        api_messages = [msg.to_api_dict() for msg in messages]
        system_text = ""
        filtered = []
        for m in api_messages:
            if m["role"] == "system":
                system_text = m["content"] if isinstance(m["content"], str) else str(m["content"])
            else:
                filtered.append(m)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": filtered,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text

        resp = await self._acall_with_retry(client.messages.create, **kwargs)

        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        return Message(role="assistant", content="".join(text_parts).strip())

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
        role: Optional[str] = None,
        tools: Optional[Mapping[str, BaseTool]] = None,
        on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> Message:
        """Send a chat completion request and return the response as a ``Message``.

        Parameters
        ----------
        role : str, optional
            Endpoint role: ``"actor"`` (default), ``"judge"``, or ``"rewrite"``.
            Determines which API endpoint, model, and temperature to use.
        model : str, optional
            Override model name (takes precedence over role-based default).

        Automatically retries on transient API errors (429, 5xx) with
        exponential backoff.
        """
        _, _, role_model, role_temp, role_extra, role_protocol = self._resolve_endpoint(role)
        model = model or role_model
        temperature = kwargs.pop("temperature", role_temp)
        client = self._get_client(role)

        # Anthropic protocol — no tool loop, direct call
        if role_protocol == "anthropic":
            return self._anthropic_generate(client, model, messages, temperature, role_extra)

        api_messages = [msg.to_api_dict() for msg in messages]
        tool_defs = None
        if tools:
            tool_defs = [
                {"type": "function", "function": tool.to_schema()}
                for tool in tools.values()
            ]

        for _ in range(_MAX_TOOL_ITERATIONS):
            request_kwargs: Dict[str, Any] = dict(kwargs)
            request_kwargs.update({
                "model": model,
                "messages": api_messages,
                "temperature": temperature,
            })
            if role_extra:
                request_kwargs.setdefault("extra_body", {})
                request_kwargs["extra_body"].update(role_extra)
            if tool_defs:
                request_kwargs["tools"] = tool_defs
                request_kwargs["tool_choice"] = "auto"

            completion = self._call_with_retry(
                client.chat.completions.create,
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

    def generate_stream(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        tools: Optional[Mapping[str, BaseTool]] = None,
        on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> Message:
        """Stream assistant text deltas and return the final assistant message.

        Preserves the existing tool-calling loop. Text deltas are emitted only
        for assistant content, not for tool argument assembly.
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
                "stream": True,
            })
            if tool_defs:
                request_kwargs["tools"] = tool_defs
                request_kwargs["tool_choice"] = "auto"

            stream = self._call_with_retry(
                self._client.chat.completions.create,
                **request_kwargs,
            )

            text_parts: List[str] = []
            tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                if delta is None:
                    continue

                content_delta = getattr(delta, "content", None)
                if content_delta:
                    text_parts.append(content_delta)
                    if on_delta:
                        on_delta(content_delta)

                for tool_delta in getattr(delta, "tool_calls", None) or []:
                    index = getattr(tool_delta, "index", 0) or 0
                    tool_call = tool_calls_by_index.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    tool_id = getattr(tool_delta, "id", None)
                    if tool_id:
                        tool_call["id"] = tool_id

                    function_delta = getattr(tool_delta, "function", None)
                    if function_delta is None:
                        continue

                    function_name = getattr(function_delta, "name", None)
                    if function_name:
                        tool_call["function"]["name"] = function_name

                    function_arguments = getattr(function_delta, "arguments", None)
                    if function_arguments:
                        tool_call["function"]["arguments"] += function_arguments

            assistant_content = "".join(text_parts)
            tool_calls = [
                tool_calls_by_index[index]
                for index in sorted(tool_calls_by_index)
            ]

            if not tool_calls:
                return Message(role="assistant", content=assistant_content)

            api_messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                tool_name = call["function"]["name"]
                tool_args_raw = call["function"]["arguments"]

                if on_tool_event:
                    on_tool_event("start", {"name": tool_name, "arguments": tool_args_raw})
                try:
                    parsed_args = json.loads(tool_args_raw or "{}")
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
                    "tool_call_id": call["id"],
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
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> Message:
        """Async version of ``generate()`` (no tool support).

        Used for parallel candidate generation and scoring in APO.
        """
        _, _, role_model, role_temp, role_extra, role_protocol = self._resolve_endpoint(role)
        model = model or role_model
        temperature = kwargs.pop("temperature", role_temp)
        async_client = self._get_async_client(role)

        # Anthropic protocol
        if role_protocol == "anthropic":
            return await self._anthropic_agenerate(async_client, model, messages, temperature, role_extra)

        api_messages = [msg.to_api_dict() for msg in messages]

        request_kwargs: Dict[str, Any] = dict(kwargs)
        request_kwargs.update({
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
        })
        if role_extra:
            request_kwargs.setdefault("extra_body", {})
            request_kwargs["extra_body"].update(role_extra)

        completion = await self._acall_with_retry(
            async_client.chat.completions.create,
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
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Message]:
        """Generate responses for multiple message lists in parallel.

        Uses ``asyncio.gather`` internally. Safe to call from sync code.
        """
        async def _run():
            tasks = [
                self.agenerate(msgs, model=model, role=role, **kwargs)
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
