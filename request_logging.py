"""Per-request disk logging for the vLLM OpenAI-compatible server.

Every inference request gets its own human-readable log file: full request
(headers/sampling params/prompt), the response written incrementally as it
is generated (so a hung/cancelled/errored request still leaves a trace), and
a performance footer. See REQUEST_LOGGING_SPEC.md (Part 1 for the format
this ports, Part 4 for this port's design) and
patches/request-logging.patch for the vLLM hook points that call into this
module.

Opt-in: disabled unless REQUEST_LOG_DIR is set in the environment (read once
per request by get_request_log_writer(), matching Part 1's
opt-in-by-default-off rule). Standalone module — importable and testable
without vLLM installed; the patch is the only thing that imports it from
inside the vendored `vllm` package.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    # Cheap upper bound before the regex pass — a client-supplied header can
    # be arbitrarily long; no need to run the regex over megabytes of it.
    text = text[: max_len * 4]
    out = _SANITIZE_RE.sub("_", text).strip("_-")
    return out[:max_len]


def _first_n_words_stub(text: str, n_words: int = 10, max_len: int = 80) -> str:
    parts: list[str] = []
    total = 0
    for word in text.split()[:n_words]:
        san = _sanitize(word, max_len - total - (1 if parts else 0))
        if not san:
            continue
        parts.append(san)
        total += len(san) + (1 if len(parts) > 1 else 0)
        if total >= max_len:
            break
    stub = "_".join(parts)[:max_len]
    return stub or "prompt"


def build_request_log_path(
    requests_dir: str | os.PathLike[str], *, prompt_text: str, request_id: str
) -> Path:
    """Same naming scheme as CachyLlama's server_request_log_build_path:
    <ISO8601 UTC>_<sanitized request id>_<first-10-words-of-prompt stub>.log,
    with -2/-3/... collision suffixes. See REQUEST_LOGGING_SPEC.md "File
    naming".
    """
    dir_path = Path(requests_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    id_part = _sanitize(request_id, 40) or "req"
    stub = _first_n_words_stub(prompt_text)

    base = f"{ts}_{id_part}_{stub}"
    candidate = dir_path / f"{base}.log"
    n = 2
    while candidate.exists():
        candidate = dir_path / f"{base}-{n}.log"
        n += 1
    return candidate


def render_chat_transcript(messages: list[dict[str, Any]]) -> str:
    """SYSTEM:/USER:/ASSISTANT:/TOOL: blocks, readable (real newlines, no
    JSON escaping). Attachments render as [image attached] etc., tool calls
    as [tool_call: name(args)]. Port of CachyLlama's
    server_request_log_render_chat_messages — same input shape (OpenAI
    `messages` array).
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role") or "unknown"
        lines.append(f"{role.upper()}:")

        content = msg.get("content")
        if content is not None:
            if isinstance(content, str):
                lines.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "text":
                        lines.append(part.get("text", ""))
                    elif ptype == "image_url":
                        lines.append("[image attached]")
                    elif ptype == "input_audio":
                        lines.append("[audio attached]")
                    elif ptype == "input_video":
                        lines.append("[video attached]")
                    elif ptype:
                        lines.append(f"[attachment: {ptype}]")
            else:
                lines.append(json.dumps(content))

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args)
                lines.append(f"[tool_call: {name}({args})]")

        lines.append("")
    return "\n".join(lines)


def _to_pretty_json(value: Any) -> str:
    """Best-effort pretty-JSON rendering for SAMPLING PARAMS / MODEL
    SETTINGS / REQUEST sections. SamplingParams is a msgspec.Struct, not a
    plain dict, so try msgspec first; fall back through a couple of other
    shapes before giving up and using repr() — this must never raise, it's
    logging.
    """
    try:
        import msgspec

        return json.dumps(msgspec.to_builtins(value), indent=2, default=str)
    except Exception:
        pass
    try:
        return json.dumps(value, indent=2, default=str)
    except Exception:
        pass
    try:
        return json.dumps(vars(value), indent=2, default=str)
    except Exception:
        return repr(value)


class _MetricsLike(Protocol):
    """Duck-typed subset of vllm.v1.metrics.stats.RequestStateStats that
    this module reads. Kept as a Protocol (rather than importing vLLM) so
    request_logging.py stays importable/testable standalone.
    """

    queued_ts: float
    scheduled_ts: float
    first_token_ts: float
    last_token_ts: float


_FooterRow = tuple[str, str]


class RequestLogWriter:
    """One instance per request. Opens the file eagerly in write_header()
    (matching Part 1: the header must be on disk even if the request never
    produces a token). All writes are flushed immediately so the file can be
    tailed live.
    """

    def __init__(self, path: Path):
        self._path = path
        self._fh = None
        self._finalized = False
        self._chunk_header_written = False
        self._thinking_forced_open = False
        self._t_start = time.monotonic()

        # "most recent chunk" vs "most recent chunk that actually carried
        # token counts" — kept separate deliberately, see
        # REQUEST_LOGGING_SPEC.md Part 1 gotcha #2 (a zeroed-out final abort
        # marker must not overwrite real partial data).
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None
        self._cached_tokens: int | None = None
        self._metrics: _MetricsLike | None = None

        self._concurrent_at_arrival: int | None = None
        self._concurrent_at_first_chunk: int | None = None
        self._got_first_chunk = False

    # -- header -----------------------------------------------------------

    def write_header(
        self,
        *,
        request_id: str,
        client_addr: str | None,
        method: str,
        path: str,
        model: str,
        resolved_model: str,
        headers: dict[str, str],
        sampling_params: Any,
        model_settings: Any,
        raw_body: str,
        prompt_text: str,
        thinking_forced_open: bool,
        concurrent_at_arrival: int | None = None,
    ) -> None:
        self._thinking_forced_open = thinking_forced_open
        self._concurrent_at_arrival = concurrent_at_arrival
        try:
            self._fh = open(self._path, "w", encoding="utf-8")
        except OSError:
            self._fh = None
            return

        w = self._fh
        w.write("\n=== VLLM REQUEST LOG ===\n")
        w.write(f"timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        w.write(f"request_id: {request_id}\n")
        w.write(f"client: {client_addr or 'unknown'}\n")
        w.write(f"method_path: {method} {path}\n")
        w.write(f"model: {model}\n")
        w.write(f"resolved_model: {resolved_model}\n")

        w.write("\n=== HEADERS ===\n")
        for k, v in headers.items():
            w.write(f"{k}: {v}\n")

        w.write("\n=== SAMPLING PARAMS ===\n")
        w.write(_to_pretty_json(sampling_params) + "\n")

        w.write("\n=== MODEL SETTINGS ===\n")
        w.write(_to_pretty_json(model_settings) + "\n")

        w.write("\n=== REQUEST ===\n")
        try:
            w.write(json.dumps(json.loads(raw_body), indent=2) + "\n")
        except Exception:
            w.write(raw_body + "\n")

        w.write("\n=== PROMPT ===\n")
        w.write(prompt_text + "\n")

        w.flush()

    # -- incremental response ---------------------------------------------

    def write_chunk(self, text: str) -> None:
        if self._fh is None or not text:
            return
        if not self._chunk_header_written:
            self._fh.write("\n=== RESPONSE (streaming) ===\n")
            if self._thinking_forced_open:
                # The chat template appended an opening "<think>" to the
                # rendered prompt, so the model's own generated tokens never
                # include it — only the matching close. Synthesize it here
                # so the log doesn't read as a truncated response. See
                # REQUEST_LOGGING_SPEC.md "Reasoning-model <think> prefix".
                self._fh.write("<think>\n")
            self._chunk_header_written = True
        self._fh.write(text)
        self._fh.flush()

    def note_progress(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
        metrics: _MetricsLike | None,
        concurrency_snapshot: int | None = None,
    ) -> None:
        if prompt_tokens is not None:
            self._prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            self._completion_tokens = completion_tokens
        if cached_tokens is not None:
            self._cached_tokens = cached_tokens
        if metrics is not None:
            self._metrics = metrics

        if not self._got_first_chunk:
            self._got_first_chunk = True
            self._concurrent_at_first_chunk = concurrency_snapshot

    # -- footer -------------------------------------------------------------

    def _build_footer_rows(self) -> list[_FooterRow]:
        rows: list[_FooterRow] = []

        if self._prompt_tokens is not None:
            rows.append(("Prompt tokens", str(self._prompt_tokens)))
        if self._completion_tokens is not None:
            rows.append(("Completion tokens", str(self._completion_tokens)))
        if self._cached_tokens is not None:
            value = str(self._cached_tokens)
            if self._prompt_tokens:
                pct = 100.0 * self._cached_tokens / self._prompt_tokens
                value += f" ({pct:.1f}%)"
            rows.append(("Cached tokens", value))

        timing = self._compute_timing()
        if timing is not None:
            (
                ttft_s,
                prefill_s,
                gen_s,
                pp_tok_s,
                tg_tok_s,
                queue_wait_s,
            ) = timing
            if pp_tok_s is not None:
                rows.append(("Prefill speed (PP)", f"{pp_tok_s:.2f} tok/s"))
            if tg_tok_s is not None:
                rows.append(("Generation speed (TG)", f"{tg_tok_s:.2f} tok/s"))
            if ttft_s is not None:
                rows.append(("Time to first token", f"{ttft_s:.3f} s"))
            if prefill_s is not None:
                rows.append(("Prefill duration", f"{prefill_s:.3f} s"))
            if gen_s is not None:
                rows.append(("Generation duration", f"{gen_s:.3f} s"))

        rows.append(("Total duration", f"{time.monotonic() - self._t_start:.3f} s"))

        if timing is not None and timing[5] is not None:
            rows.append(("Queue wait", f"{timing[5]:.3f} s"))

        if self._concurrent_at_arrival is not None:
            rows.append(
                ("Concurrent @ prefill start", f"{self._concurrent_at_arrival} other request(s)")
            )
        if self._concurrent_at_first_chunk is not None:
            rows.append(
                ("Concurrent @ decode start", f"{self._concurrent_at_first_chunk} other request(s)")
            )

        return rows

    def _compute_timing(
        self,
    ) -> tuple[
        float | None, float | None, float | None, float | None, float | None, float | None
    ] | None:
        """Returns (ttft_s, prefill_s, generation_s, pp_tok_s, tg_tok_s,
        queue_wait_s), each None individually when the underlying
        timestamps aren't available. Uses vLLM's own
        RequestStateStats timestamps (queued_ts/scheduled_ts/first_token_ts/
        last_token_ts) directly rather than reimplementing the math — see
        REQUEST_LOGGING_SPEC.md Part 4 data-mapping table. Prefill duration
        and TTFT are the same interval (scheduled_ts -> first_token_ts) in
        vLLM's V1 engine, same as CachyLlama's port; both rows are kept per
        Part 1's file format.
        """
        m = self._metrics
        if m is None:
            return None

        queued_ts = getattr(m, "queued_ts", 0.0) or 0.0
        scheduled_ts = getattr(m, "scheduled_ts", 0.0) or 0.0
        first_token_ts = getattr(m, "first_token_ts", 0.0) or 0.0
        last_token_ts = getattr(m, "last_token_ts", 0.0) or 0.0

        ttft_s: float | None = None
        prefill_s: float | None = None
        gen_s: float | None = None
        pp_tok_s: float | None = None
        tg_tok_s: float | None = None
        queue_wait_s: float | None = None

        if scheduled_ts > 0 and first_token_ts > 0:
            ttft_s = first_token_ts - scheduled_ts
            prefill_s = ttft_s
            if ttft_s > 0 and self._prompt_tokens:
                pp_tok_s = self._prompt_tokens / ttft_s

        if first_token_ts > 0 and last_token_ts > 0 and self._completion_tokens and self._completion_tokens > 1:
            gen_s = last_token_ts - first_token_ts
            if gen_s > 0:
                tg_tok_s = (self._completion_tokens - 1) / gen_s

        if queued_ts > 0 and scheduled_ts > 0:
            queue_wait_s = scheduled_ts - queued_ts

        if ttft_s is None and prefill_s is None and gen_s is None and queue_wait_s is None:
            return None
        return ttft_s, prefill_s, gen_s, pp_tok_s, tg_tok_s, queue_wait_s

    def _write_footer_block(self, label: str) -> None:
        if self._fh is None:
            return
        rows = self._build_footer_rows()
        self._fh.write(f"\n=== {label} ===\n")
        if rows:
            width = max(len(r[0]) for r in rows)
            for name, value in rows:
                self._fh.write(f"{name.ljust(width)} : {value}\n")
        self._fh.flush()

    def write_footer(self, *, label: str = "PERFORMANCE") -> None:
        if self._finalized or self._fh is None:
            return
        self._finalized = True
        self._write_footer_block(label)

    def write_error(self, exc: BaseException, *, http_status: int | None = None) -> None:
        if self._finalized or self._fh is None:
            return
        self._finalized = True
        w = self._fh
        w.write("\n=== ERROR ===\n")
        w.write(f"exception_type: {type(exc).__name__}\n")
        w.write(f"message: {exc}\n")
        if http_status is not None:
            w.write(f"http_status: {http_status}\n")
        w.write("traceback:\n")
        w.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        w.flush()
        self._write_footer_block("PERFORMANCE (partial)")

    def close(self) -> None:
        if self._fh is None:
            return
        if not self._finalized:
            # Reached without an explicit write_footer()/write_error() call —
            # e.g. a cancelled generator whose `finally` block only calls
            # close(). Best-effort partial footer, same as CachyLlama's RAII
            # destructor safety net (REQUEST_LOGGING_SPEC.md Part 1 gotcha #1).
            self._finalized = True
            self._fh.write("\n=== CANCELLED ===\n")
            self._fh.write(
                "message: request was destroyed before completion "
                "(client disconnect or server-side cancellation)\n"
            )
            self._write_footer_block("PERFORMANCE (partial)")
        try:
            self._fh.flush()
            self._fh.close()
        except OSError:
            pass
        self._fh = None


class NullRequestLogWriter:
    """Every method a no-op. Returned by get_request_log_writer() when
    REQUEST_LOG_DIR is unset, so call sites in the vLLM patch never branch
    on the setting themselves.
    """

    def write_header(self, **kwargs: Any) -> None:
        pass

    def write_chunk(self, text: str) -> None:
        pass

    def note_progress(self, **kwargs: Any) -> None:
        pass

    def write_footer(self, *, label: str = "PERFORMANCE") -> None:
        pass

    def write_error(self, exc: BaseException, *, http_status: int | None = None) -> None:
        pass

    def close(self) -> None:
        pass


def get_request_log_writer(request_id: str, prompt_text: str) -> RequestLogWriter | NullRequestLogWriter:
    """Reads REQUEST_LOG_DIR from the environment itself (Part 4's "Config"
    decision — a plain env var, not a registered vllm.envs.py knob). Unset
    or empty => logging is off for this request.
    """
    requests_dir = os.environ.get("REQUEST_LOG_DIR")
    if not requests_dir:
        return NullRequestLogWriter()
    try:
        path = build_request_log_path(requests_dir, prompt_text=prompt_text, request_id=request_id)
        return RequestLogWriter(path)
    except OSError:
        return NullRequestLogWriter()
