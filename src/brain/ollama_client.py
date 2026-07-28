"""
Asynchronous client for the Ollama REST API.

OllamaClient is pure transport: it exposes generate, chat, streaming,
model listing, model switching, preloading, and health checks against a
local Ollama server. It contains no decision logic. The blocking
``requests`` library is wrapped in worker threads so every public method
is fully async.

Timeout design
--------------
With ``stream: false`` the server sends nothing until generation is
complete, so a single HTTP read timeout covers the *entire* generation.
An unbounded generation therefore always presents as a read timeout
rather than as the overlong generation it actually is. Three mechanisms
address that:

* ``num_predict`` bounds how many tokens the model may emit.
* ``think`` disables reasoning preambles on thinking models, which
  otherwise emit hundreds of tokens before the answer.
* ``format`` constrains output to a JSON schema, so the model cannot
  wander into prose.

Connect and read timeouts are configured separately: a server that is
down should fail in a second, while a legitimate long generation is
allowed its full budget.

Every request records :class:`RequestMetrics` from the server's own
counters (``prompt_eval_count``, ``eval_count``, ``load_duration``) so
slow calls can be attributed to prompt size, output size, or model
loading rather than guessed at.

API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Sequence

import requests
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.exceptions import BrainError
from src.logger import get_logger

_NANOSECONDS_PER_SECOND = 1_000_000_000


class ChatMessage(BaseModel):
    """
    One message in an Ollama chat conversation.

    Attributes:
        role: Message role: ``system``, ``user``, or ``assistant``.
        content: Message text.
    """

    role: str = Field(min_length=1)
    content: str


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """
    Timing and token accounting for one Ollama request.

    Durations reported by the server are in nanoseconds and are converted
    to seconds here. Fields are optional because older servers and some
    endpoints omit individual counters.

    Attributes:
        operation: Caller-supplied label, for example ``classify``.
        model: Model that served the request.
        wall_seconds: Client-observed latency including HTTP overhead.
        prompt_chars: Length of the submitted prompt in characters.
        prompt_tokens: Server-reported prompt token count.
        response_tokens: Server-reported generated token count.
        load_seconds: Time the server spent loading the model.
        eval_seconds: Time the server spent generating.
        truncated: True when generation stopped because it hit the token
            cap rather than finishing naturally.
    """

    operation: str
    model: str
    wall_seconds: float
    prompt_chars: int
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    load_seconds: Optional[float] = None
    eval_seconds: Optional[float] = None
    truncated: bool = False

    @property
    def tokens_per_second(self) -> Optional[float]:
        """
        Return the generation rate, when it can be computed.

        Returns:
            Tokens per second, or None when counters are unavailable.
        """
        if not self.response_tokens or not self.eval_seconds:
            return None
        return self.response_tokens / self.eval_seconds

    def describe(self) -> str:
        """
        Render a single-line human-readable summary.

        Returns:
            Log-friendly description of the request.
        """
        parts = [
            f"op={self.operation}",
            f"model={self.model}",
            f"wall={self.wall_seconds:.2f}s",
            f"prompt_chars={self.prompt_chars}",
        ]
        if self.prompt_tokens is not None:
            parts.append(f"prompt_tokens={self.prompt_tokens}")
        if self.response_tokens is not None:
            parts.append(f"response_tokens={self.response_tokens}")
        if self.load_seconds:
            parts.append(f"load={self.load_seconds:.2f}s")
        rate = self.tokens_per_second
        if rate is not None:
            parts.append(f"rate={rate:.1f}tok/s")
        if self.truncated:
            parts.append("TRUNCATED")
        return " ".join(parts)


class OllamaClientConfig(BaseModel):
    """
    Configuration for the Ollama client.

    Attributes:
        host: Base URL of the Ollama server.
        model: Default model name.
        timeout_seconds: Read timeout for long inference calls.
        fast_timeout_seconds: Read timeout for bounded structured calls
            such as classification, which must fail fast rather than
            occupy the full inference budget.
        connect_timeout_seconds: TCP connect timeout. Kept short so an
            unreachable server fails immediately.
        health_timeout_seconds: Request timeout for health checks.
        num_ctx: Context window size passed to the model.
        keep_alive: How long the server keeps the model resident after a
            request. Avoids paying model load cost on every call.
        think: Whether thinking models may emit a reasoning preamble.
            Disabled by default because it multiplies latency for tasks
            whose answer is a few tokens of JSON.
        max_retries: Attempts per request before failing.
    """

    host: str = Field(default="http://localhost:11434", min_length=1)
    model: str = Field(default="qwen3:4b", min_length=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    fast_timeout_seconds: float = Field(default=25.0, gt=0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    health_timeout_seconds: float = Field(default=5.0, gt=0)
    num_ctx: int = Field(default=4096, gt=0)
    keep_alive: str = Field(default="10m")
    think: bool = Field(default=False)
    max_retries: int = Field(default=2, ge=1)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        **overrides: object,
    ) -> "OllamaClientConfig":
        """
        Build client configuration from application settings.

        Args:
            settings: Existing application settings object.
            **overrides: Explicit values replacing setting defaults.

        Returns:
            Ollama client configuration.
        """
        values: dict[str, object] = {
            "host": settings.ollama_host,
            "model": settings.ollama_model,
            "timeout_seconds": settings.ollama_timeout,
            "fast_timeout_seconds": settings.ollama_fast_timeout,
            "connect_timeout_seconds": settings.ollama_connect_timeout,
            "num_ctx": settings.ollama_num_ctx,
            "keep_alive": settings.ollama_keep_alive,
            "think": settings.ollama_think,
            "max_retries": settings.max_retries,
        }
        values.update(overrides)
        return cls(**values)


class OllamaClient:
    """
    Async client for a local Ollama server.

    The active model can be switched at runtime; individual calls may also
    override the model per request. All failures are normalized into
    BrainError so callers handle one exception type.
    """

    def __init__(
        self,
        config: Optional[OllamaClientConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            config: Optional client configuration.
            settings: Optional application settings used when config is
                absent.
        """
        self._settings = settings or get_settings()
        self._config = config or OllamaClientConfig.from_settings(
            self._settings
        )
        self._model = self._config.model
        self._logger = get_logger(__name__)
        self._last_metrics: Optional[RequestMetrics] = None

    @property
    def host(self) -> str:
        """
        Return the Ollama server base URL.

        Returns:
            Server base URL.
        """
        return self._config.host.rstrip("/")

    @property
    def model(self) -> str:
        """
        Return the active model name.

        Returns:
            Active model name.
        """
        return self._model

    @property
    def config(self) -> OllamaClientConfig:
        """
        Return the active client configuration.

        Returns:
            Client configuration.
        """
        return self._config

    @property
    def last_metrics(self) -> Optional[RequestMetrics]:
        """
        Return metrics for the most recent completed request.

        Returns:
            Metrics, or None when no request has completed.
        """
        return self._last_metrics

    def set_model(self, model: str) -> None:
        """
        Switch the active model.

        Args:
            model: Model name to use for subsequent calls.

        Raises:
            BrainError: If the model name is empty.
        """
        if not model or not model.strip():
            raise BrainError(
                "Model name cannot be empty",
                code="OLLAMA_MODEL_EMPTY",
            )
        self._model = model.strip()
        self._logger.info("Switched Ollama model to '%s'", self._model)

    async def resolve_model(self) -> str:
        """
        Ensure the active model exists on the server, falling back if not.

        Resolution order:

        1. Exact match of the active model name — kept as is.
        2. Same base name with a different tag (for example configured
           ``qwen3`` resolving to installed ``qwen3:4b``) — switched to,
           with a warning.
        3. Otherwise the first installed model — switched to, with a
           warning.

        When the server's model list cannot be read or is empty, the
        configured model is kept and a warning is logged; the method
        never raises for resolution problems.

        Returns:
            The active model name after resolution.
        """
        try:
            installed = await self.list_models()
        except BrainError as exc:
            self._logger.warning(
                "Could not list Ollama models (%s); keeping configured "
                "model '%s'",
                exc,
                self._model,
            )
            return self._model

        if not installed:
            self._logger.warning(
                "Ollama reports no installed models; keeping configured "
                "model '%s' (requests will fail until a model is pulled)",
                self._model,
            )
            return self._model

        if self._model in installed:
            return self._model

        base_name = self._model.split(":", 1)[0]
        replacement = next(
            (
                name
                for name in installed
                if name.split(":", 1)[0] == base_name
            ),
            installed[0],
        )

        self._logger.warning(
            "Configured Ollama model '%s' is not installed; falling back "
            "to '%s' (installed models: %s)",
            self._model,
            replacement,
            ", ".join(installed),
        )
        self.set_model(replacement)
        return self._model

    async def health_check(self) -> bool:
        """
        Check whether the Ollama server is reachable.

        Returns:
            True when the server responds to a version request.
        """
        def probe() -> bool:
            response = requests.get(
                f"{self.host}/api/version",
                timeout=self._config.health_timeout_seconds,
            )
            response.raise_for_status()
            return True

        try:
            return await asyncio.to_thread(probe)
        except Exception as exc:
            self._logger.warning("Ollama health check failed: %s", exc)
            return False

    async def list_models(self) -> list[str]:
        """
        List models available on the Ollama server.

        Returns:
            Model names.

        Raises:
            BrainError: If the server cannot be queried.
        """
        def fetch() -> list[str]:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=self._config.health_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            return [
                str(model.get("name", ""))
                for model in models
                if model.get("name")
            ]

        try:
            return await asyncio.to_thread(fetch)
        except Exception as exc:
            raise BrainError(
                f"Failed to list Ollama models: {exc}",
                code="OLLAMA_LIST_MODELS_FAILED",
            ) from exc

    async def preload(self) -> bool:
        """
        Load the active model into server memory ahead of first use.

        An empty prompt instructs Ollama to load the model without
        generating. Doing this at startup moves multi-second model load
        cost out of the first user request, where it would otherwise be
        indistinguishable from a slow generation.

        Returns:
            True when the model was loaded successfully.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "keep_alive": self._config.keep_alive,
        }

        started = time.monotonic()
        try:
            await self._post_json(
                "/api/generate",
                payload,
                operation="preload",
                read_timeout=self._config.timeout_seconds,
            )
        except BrainError as exc:
            self._logger.warning(
                "Failed to preload model '%s': %s", self._model, exc
            )
            return False

        self._logger.info(
            "Preloaded model '%s' in %.2fs",
            self._model,
            time.monotonic() - started,
        )
        return True

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        num_predict: Optional[int] = None,
        response_format: Optional[dict[str, Any] | str] = None,
        stop: Optional[Sequence[str]] = None,
        timeout_seconds: Optional[float] = None,
        operation: str = "generate",
    ) -> str:
        """
        Run one-shot text generation.

        Args:
            prompt: Prompt text.
            system: Optional system prompt.
            model: Optional model override.
            options: Optional raw Ollama options, merged last so callers
                retain full control.
            num_predict: Maximum tokens to generate. Strongly recommended
                for short structured answers; without it generation is
                unbounded and read timeouts become likely.
            response_format: ``"json"`` or a JSON schema constraining the
                output shape.
            stop: Optional stop sequences.
            timeout_seconds: Read timeout override for this call.
            operation: Label used in timing logs and metrics.

        Returns:
            Generated text.

        Raises:
            BrainError: If generation fails.
        """
        payload: dict[str, Any] = {
            "model": model or self._model,
            "prompt": prompt,
            "stream": False,
            "think": self._config.think,
            "keep_alive": self._config.keep_alive,
            "options": self._build_options(options, num_predict, stop),
        }
        if system is not None:
            payload["system"] = system
        if response_format is not None:
            payload["format"] = response_format

        data = await self._post_json(
            "/api/generate",
            payload,
            operation=operation,
            read_timeout=timeout_seconds,
            prompt_chars=len(prompt) + len(system or ""),
        )
        return str(data.get("response", ""))

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        num_predict: Optional[int] = None,
        response_format: Optional[dict[str, Any] | str] = None,
        stop: Optional[Sequence[str]] = None,
        timeout_seconds: Optional[float] = None,
        operation: str = "chat",
    ) -> str:
        """
        Run a chat completion over a message history.

        Args:
            messages: Conversation messages including system prompts.
            model: Optional model override.
            options: Optional raw Ollama options, merged last.
            num_predict: Maximum tokens to generate.
            response_format: ``"json"`` or a JSON schema.
            stop: Optional stop sequences.
            timeout_seconds: Read timeout override for this call.
            operation: Label used in timing logs and metrics.

        Returns:
            Assistant reply text.

        Raises:
            BrainError: If the chat call fails.
        """
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "think": self._config.think,
            "keep_alive": self._config.keep_alive,
            "options": self._build_options(options, num_predict, stop),
        }
        if response_format is not None:
            payload["format"] = response_format

        data = await self._post_json(
            "/api/chat",
            payload,
            operation=operation,
            read_timeout=timeout_seconds,
            prompt_chars=sum(len(m.content) for m in messages),
        )
        message = data.get("message", {})
        if not isinstance(message, dict):
            raise BrainError(
                "Ollama chat returned an unexpected payload shape",
                code="OLLAMA_CHAT_MALFORMED",
            )
        return str(message.get("content", ""))

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        num_predict: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream one-shot generation token chunks.

        Streaming resets the HTTP read timeout on every chunk, so long
        generations do not trip a read timeout the way a single
        non-streaming request does.

        Args:
            prompt: Prompt text.
            system: Optional system prompt.
            model: Optional model override.
            options: Optional raw Ollama options.
            num_predict: Maximum tokens to generate.

        Yields:
            Response text chunks as they are produced.

        Raises:
            BrainError: If streaming fails.
        """
        payload: dict[str, Any] = {
            "model": model or self._model,
            "prompt": prompt,
            "stream": True,
            "think": self._config.think,
            "keep_alive": self._config.keep_alive,
            "options": self._build_options(options, num_predict, None),
        }
        if system is not None:
            payload["system"] = system

        async for chunk in self._stream("/api/generate", payload):
            text = chunk.get("response")
            if text:
                yield str(text)

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        num_predict: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion chunks over a message history.

        Args:
            messages: Conversation messages including system prompts.
            model: Optional model override.
            options: Optional raw Ollama options.
            num_predict: Maximum tokens to generate.

        Yields:
            Assistant reply text chunks as they are produced.

        Raises:
            BrainError: If streaming fails.
        """
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": [message.model_dump() for message in messages],
            "stream": True,
            "think": self._config.think,
            "keep_alive": self._config.keep_alive,
            "options": self._build_options(options, num_predict, None),
        }

        async for chunk in self._stream("/api/chat", payload):
            message = chunk.get("message")
            if isinstance(message, dict) and message.get("content"):
                yield str(message["content"])

    def _build_options(
        self,
        options: Optional[dict[str, Any]],
        num_predict: Optional[int],
        stop: Optional[Sequence[str]],
    ) -> dict[str, Any]:
        """
        Assemble Ollama model options for a request.

        Caller-supplied ``options`` are merged last so they always win.

        Args:
            options: Optional raw options from the caller.
            num_predict: Optional generation cap.
            stop: Optional stop sequences.

        Returns:
            Options dictionary for the request payload.
        """
        merged: dict[str, Any] = {"num_ctx": self._config.num_ctx}
        if num_predict is not None:
            merged["num_predict"] = num_predict
        if stop:
            merged["stop"] = list(stop)
        if options:
            merged.update(options)
        return merged

    def _record_metrics(
        self,
        data: dict[str, Any],
        operation: str,
        wall_seconds: float,
        prompt_chars: int,
    ) -> RequestMetrics:
        """
        Extract and log metrics from a completed response.

        Args:
            data: Parsed response payload.
            operation: Caller-supplied operation label.
            wall_seconds: Client-observed latency.
            prompt_chars: Submitted prompt length in characters.

        Returns:
            Recorded metrics.
        """
        def seconds(key: str) -> Optional[float]:
            raw = data.get(key)
            if isinstance(raw, (int, float)) and raw > 0:
                return float(raw) / _NANOSECONDS_PER_SECOND
            return None

        metrics = RequestMetrics(
            operation=operation,
            model=str(data.get("model", self._model)),
            wall_seconds=wall_seconds,
            prompt_chars=prompt_chars,
            prompt_tokens=data.get("prompt_eval_count"),
            response_tokens=data.get("eval_count"),
            load_seconds=seconds("load_duration"),
            eval_seconds=seconds("eval_duration"),
            truncated=data.get("done_reason") == "length",
        )
        self._last_metrics = metrics

        self._logger.info("Ollama request complete: %s", metrics.describe())
        if metrics.truncated:
            self._logger.warning(
                "Ollama generation for '%s' hit the token cap; the answer "
                "may be incomplete",
                operation,
            )
        return metrics

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        operation: str = "request",
        read_timeout: Optional[float] = None,
        prompt_chars: int = 0,
    ) -> dict[str, Any]:
        """
        POST a JSON payload and return the parsed JSON response.

        Retries transient failures. Read timeouts are *not* retried by
        default beyond the configured attempt count, because a timeout
        caused by unbounded generation will recur deterministically and
        retrying only multiplies the delay.

        Args:
            path: API path.
            payload: JSON request payload.
            operation: Label used in timing logs.
            read_timeout: Read timeout override in seconds.
            prompt_chars: Submitted prompt length, for metrics.

        Returns:
            Parsed response object.

        Raises:
            BrainError: If the request or parsing fails.
        """
        url = f"{self.host}{path}"
        timeout = (
            self._config.connect_timeout_seconds,
            read_timeout or self._config.timeout_seconds,
        )

        def call() -> dict[str, Any]:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise ValueError(self._describe_http_error(response))
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Response payload is not a JSON object")
            return data

        last_error: Optional[Exception] = None
        for attempt in range(1, self._config.max_retries + 1):
            started = time.monotonic()
            self._logger.debug(
                "Ollama %s starting (attempt %d/%d, prompt_chars=%d, "
                "read_timeout=%.1fs)",
                operation,
                attempt,
                self._config.max_retries,
                prompt_chars,
                timeout[1],
            )
            try:
                data = await asyncio.to_thread(call)
            except requests.exceptions.ReadTimeout as exc:
                elapsed = time.monotonic() - started
                self._logger.warning(
                    "Ollama %s read-timed-out after %.1fs. The model was "
                    "still generating. Bound the output with num_predict, "
                    "disable thinking, or raise the timeout.",
                    operation,
                    elapsed,
                )
                raise BrainError(
                    f"Ollama {operation} exceeded the {timeout[1]:.0f}s read "
                    f"timeout while generating. This usually means the "
                    f"output was not bounded, not that the server is slow.",
                    code="OLLAMA_READ_TIMEOUT",
                ) from exc
            except (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                last_error = exc
                self._logger.warning(
                    "Ollama %s could not reach %s (attempt %d/%d): %s",
                    operation,
                    url,
                    attempt,
                    self._config.max_retries,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    "Ollama %s failed (attempt %d/%d): %s",
                    operation,
                    attempt,
                    self._config.max_retries,
                    exc,
                )
            else:
                self._record_metrics(
                    data,
                    operation=operation,
                    wall_seconds=time.monotonic() - started,
                    prompt_chars=prompt_chars,
                )
                return data

            if attempt < self._config.max_retries:
                await asyncio.sleep(self._settings.retry_delay * attempt)

        raise BrainError(
            f"Ollama request to {path} failed after "
            f"{self._config.max_retries} attempt(s): {last_error}",
            code="OLLAMA_REQUEST_FAILED",
        ) from last_error

    async def _stream(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """
        POST a JSON payload and yield streamed JSON-lines chunks.

        A background thread reads the blocking HTTP stream and feeds an
        asyncio queue, keeping the event loop unblocked.

        Args:
            path: API path.
            payload: JSON request payload.

        Yields:
            Parsed JSON chunk objects.

        Raises:
            BrainError: If the streaming request fails.
        """
        url = f"{self.host}{path}"
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[object]] = asyncio.Queue()
        timeout = (
            self._config.connect_timeout_seconds,
            self._config.timeout_seconds,
        )

        def reader() -> None:
            try:
                with requests.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=timeout,
                ) as response:
                    if response.status_code >= 400:
                        raise ValueError(
                            self._describe_http_error(response)
                        )
                    for line in response.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line.decode("utf-8"))
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:  # delivered to the async consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        reader_task = asyncio.create_task(asyncio.to_thread(reader))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise BrainError(
                        f"Ollama streaming request to {path} failed: {item}",
                        code="OLLAMA_STREAM_FAILED",
                    ) from item
                if isinstance(item, dict):
                    yield item
        finally:
            await reader_task

    @staticmethod
    def _describe_http_error(response: requests.Response) -> str:
        """
        Build a diagnostic message from an Ollama error response.

        Ollama returns errors as ``{"error": "..."}`` JSON bodies (for
        example HTTP 404 with ``model '<name>' not found`` when the
        requested model is not installed). Surfacing that body is
        essential: a bare status code makes a missing model look like a
        missing endpoint.

        Args:
            response: HTTP response with a >= 400 status code.

        Returns:
            Human-readable error description including the server detail.
        """
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("error", "")).strip()
        except Exception:
            detail = response.text[:200].strip()

        message = f"HTTP {response.status_code} {response.reason}"
        if detail:
            message = f"{message}: {detail}"
        return message


__all__ = [
    "ChatMessage",
    "OllamaClient",
    "OllamaClientConfig",
    "RequestMetrics",
]
