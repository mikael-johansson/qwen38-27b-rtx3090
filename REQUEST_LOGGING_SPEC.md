# Full request/response disk logging — spec & porting guide

This document has two parts:

- **Part 1 — Portable spec.** What the feature does and why, independent of any
  specific server implementation. Read this first regardless of which backend
  you're porting to.
- **Part 2 — oMLX reference implementation.** Exactly what was built in this
  repo, as a concrete example to copy from.
- **Part 3 — CachyLlama porting plan.** File:line-grounded plan for
  implementing the same feature in `~/git/dl/CachyLLama` (a llama.cpp fork).

Original one-paragraph ask, for reference, is in `REQUEST_LOGGING.md` at the
repo root.

---

## Part 1 — Portable spec

### Goal

Every inference request gets its own human-readable log file on disk,
containing:

1. The full request — HTTP method/path/headers, model, resolved sampling
   params, and the raw request body — plus a readable (non-JSON-escaped)
   rendering of the prompt/chat transcript.
2. The full response, **written incrementally as tokens are generated** —
   not just the final result — so a request that hangs, gets cancelled, or
   errors out mid-generation still leaves a readable trace of what was
   actually produced.
3. A performance footer: prompt/completion/cached token counts, PP tok/s,
   TG tok/s, TTFT, queue-wait time, and whether prefill/decode ran
   concurrently with other requests.
4. On error or cancellation: an error block (exception type, message,
   traceback) **plus a best-effort partial performance footer** using
   whatever was generated before the failure — never just silence.

### Why "incremental, not just final" is the whole point

The feature exists to answer one question after the fact: *what actually
happened to this request?* A design that only writes a log entry when the
request finishes successfully answers that question for the boring case and
is silent for exactly the interesting one. Three real incidents drove this
during the oMLX implementation, each fixed for a reason worth carrying into
any port:

1. **Client disconnects/timeouts** (very common with long generations and
   agentic-gateway clients that retry). Whatever your language's
   "the consumer went away" signal is (`asyncio.CancelledError` in Python; a
   cooperative should-stop flag + task-cancel in a C++ queue-based server;
   a context-cancel in Go), it must be treated as a **first-class case that
   still produces a log entry** — not swallowed as "eh, nobody's listening
   anymore." Whoever operates the server is the one who needs the log, not
   the disconnected client.
2. **A "final abort" event can carry zeroed-out counters that overwrite good
   partial data.** A memory-guard kill, a context-length-exceeded abort, or
   any other terminal error event is usually represented as one more
   message/output object in the same stream as normal chunks — and it often
   carries no token counts (because there's nothing more to report). If your
   code naively does "last output wins" for building the partial-performance
   footer, that final zero-valued marker **overwrites the real token count
   from the last successful chunk**, and your partial footer reports
   `completion_tokens: 0` after 700 words of real, disk-logged output. Track
   "last output that actually carried token data" **separately** from "most
   recent output," and prefer the former for the footer.
3. **If your backend has separate atomic (non-streaming) and incremental
   (streaming) generation code paths, route the non-streaming HTTP case
   through the incremental path internally too**, and just aggregate the
   chunks yourself before replying with a single JSON body. Otherwise the
   very requests most likely to need debugging — long ones that end up
   cancelled or erroring out — are exactly the ones your atomic call gives
   you zero visibility into, because the single await/blocking call never
   returns anything until it's fully done (or fully failed with nothing).
   **Check first** whether your backend already generates incrementally
   internally regardless of HTTP streaming mode (many do — task-queue /
   slot-based servers like llama.cpp already work this way) — if so, this
   requirement is free and you just need to hook the existing incremental
   path.

### File naming

```
<ISO8601 UTC timestamp>_<request id>_<first 10 words of prompt, sanitized>.log
```

- Timestamp: filesystem-safe ISO8601, e.g. `2026-08-12T07-30-21Z` (colons
  replaced).
- Request id: the client-supplied request-id header if present, otherwise a
  generated id. Lets you grep by id when correlating with client-side logs.
  Capped at ~40 chars and sanitized (client-supplied headers are untrusted
  length/content).
- Prompt stub: first 10 words of the prompt (completions) or the last user
  message's text (chat), sanitized to `[A-Za-z0-9_-]`, truncated to ~80
  chars. Falls back to an attachment filename (if the request carries a
  named file) or an `OCR_<model>` tag when there's no usable text at all
  (image-only vision requests).
- Collision guard: if the exact filename already exists (two requests in the
  same second), append `-2`, `-3`, ...

### File format

Plain text, sections delimited by `=== SECTION NAME ===` markers, written
incrementally (each write flushed immediately so the file can be `tail -f`'d
live). The file starts with a single leading blank line before the first
marker (purely cosmetic — makes the file read cleanly when concatenated or
viewed after other output). Exact section order:

```

=== <BACKEND> REQUEST LOG ===
timestamp: <ISO8601>
request_id: <id>
client: <ip or 'unknown'>
method_path: <METHOD> <path>
model: <requested model>
resolved_model: <resolved/alias-expanded model>

=== HEADERS ===
<header-name>: <value>
... (verbatim, one per line — see "Header redaction" below)

=== SAMPLING PARAMS ===
<effective sampling params as pretty JSON — after per-model-default
resolution, not just what the client sent>

=== MODEL SETTINGS ===
<per-model server-side settings/overrides, if any, as pretty JSON>

=== REQUEST ===
<raw request body as pretty JSON — exactly what the client sent>

=== PROMPT ===
<readable, un-escaped rendering of the prompt/chat — real newlines, not
\n escape sequences. For chat: SYSTEM:/USER:/ASSISTANT:/TOOL: blocks with
attachments rendered as [image attached] / [file attached: name.pdf] and
tool calls as [tool_call: name(args)]>

=== RESPONSE (streaming) ===
<raw generated text, appended chunk by chunk as generated — including any
synthetic reasoning-block opener; see "Reasoning-model <think> prefix"
below>
```

Then exactly one of:

```
=== PERFORMANCE ===
Prompt tokens               : N
Completion tokens           : N
Cached tokens                : N (NN.N%)
Cache hit rate (all-time)     : NN.N%                (only if available)
Prefill speed (PP)            : NNN.NN tok/s
Generation speed (TG)         : NNN.NN tok/s
Time to first token           : N.NNN s
Model load                    : N.NNN s              (only if non-trivial)
Prefill duration              : N.NNN s
Generation duration           : N.NNN s
Total duration                : N.NNN s
Queue wait                    : N.NNN s              (only if measurable)
Actual prefill duration       : N.NNN s              (only if measurable)
Concurrent @ prefill start    : N other request(s)
Concurrent @ decode start     : N other request(s)
```

Rows are `(label, value-with-unit)` pairs rendered as a table: the label
column is left-justified to the width of the longest *included* label in
that particular footer (so it re-aligns cleanly whichever optional rows end
up present), followed by ` : ` and the value with its unit inline (`tok/s`,
`s`, `%`, `other request(s)`) — never a bare unitless number. This is a
deliberate readability requirement, not just cosmetic: a log meant to be
read by a human during an incident should not require the reader to
remember which field is in which unit.

or, on failure:

```
=== ERROR ===
exception_type: <type name>
message: <message>
http_status: <code>       (if applicable)
detail: <detail>          (if applicable)
traceback:
<full traceback/stack trace>

=== PERFORMANCE (partial) ===
<same table as above, but only the rows that are actually computable from
whatever was generated before the failure — omit rows you don't have data
for, never fake them as 0>
```

Fields that can't be computed are **omitted**, not filled with `0` — a
missing `Queue wait` row means "we don't know," a present `Queue wait :
0.000 s` row means "we measured zero." This distinction matters for anyone
reading the log later.

### Reasoning-model `<think>` prefix

Reasoning models (Qwen3-style, DeepSeek-R1-style, etc.) commonly have a chat
template that appends an opening `<think>` tag to the *end of the rendered
prompt*, forcing the model straight into its reasoning phase. That means the
literal opening tag is a **prompt/template artifact**, not a generated
token — the model's own output typically starts with reasoning text and
later emits the *closing* `</think>` tag, but never generates the opening
one itself. Some schedulers additionally synthesize/echo that opening tag
back as the first chunk of "generated" output for API-completeness (oMLX's
does, for parity with clients that expect to see it), then strip it back out
of the client-facing text in some code paths for protocol-correctness
reasons specific to that endpoint (e.g. avoiding a visible duplicate when a
raw-completions client's own prompt already visually ends in `<think>`).

**For the request log specifically** (as opposed to the client-facing
response), always render the opening `<think>` tag at the start of the
`=== RESPONSE ===` section when the request is known to have started in
thinking mode — regardless of whether your backend's client-facing protocol
strips it. The log is not a protocol payload and has no duplication concern;
showing `<think>...reasoning...</think>answer` in full is unambiguously more
readable than showing only the trailing `</think>` with no matching opener,
which reads as a truncation bug even when it isn't one.

### Configuration decisions (carry forward deliberately, don't assume)

These were explicit product decisions made for oMLX, not universal truths —
re-confirm them for each new backend/deployment rather than copying blindly:

- **Opt-in, off by default.** Full prompts/responses/headers on disk is
  sensitive and unbounded disk growth; an operator must explicitly enable
  it.
- **Headers logged verbatim, no redaction** — including `Authorization`.
  This was an explicit choice favoring debuggability over defense-in-depth
  (the log directory is assumed to be at least as protected as the server
  itself). Reconsider this if the new deployment's threat model differs.
- **No retention/auto-pruning.** Per-request log files are kept forever
  (unlike the general server log, which rotates). Operators manage disk
  space manually.
- **Filename collisions**, not request overwrites — never silently drop a
  log entry because two requests landed in the same second.

### Core data needed per request

To fill in the performance footer you need, at minimum, access to (names
below are the oMLX/Python terms — see Part 3 for the llama.cpp equivalents):

| Concept | Used for |
|---|---|
| `arrival_time` (when the request was accepted into the scheduler/queue) | queue wait |
| `prefill_started_at` (when it actually started processing, left the queue) | queue wait, prefill duration |
| `generation_started_at` (when the first output token was produced) | prefill duration, TTFT |
| `prompt_tokens`, `completion_tokens`, `cached_tokens` (cumulative, not per-chunk deltas) | throughput, cache hit ratio |
| count of other requests admitted/processing when this one started prefill / started decode | concurrency visibility |
| whatever your scheduler calls "prefix cache hit" / "reused tokens" | `cached_tokens` |

If any of these aren't cheaply available in your backend, omit the derived
fields rather than faking them — see "fields that can't be computed" above.

---

## Part 2 — oMLX reference implementation

Concrete implementation in this repo, for cross-referencing intent when the
description in Part 1 is ambiguous.

### Settings

`omlx/settings.py` — `LoggingSettings.request_logging_enabled: bool = False`,
plus `LoggingSettings.get_requests_dir(base_path)` →
`{log_dir}/requests`.

### Core module: `omlx/request_logging.py`

- `build_request_log_path(requests_dir, *, prompt=None, messages=None, model="", request_id="")`
  — filename construction (see "File naming" above). Also exports
  `format_chat_transcript(messages)` for the readable `=== PROMPT ===`
  section, used both here and directly by the writer.
- `RequestLogWriter` — one instance per request, lazily opens the file on
  first write:
  - `write_header(...)` — writes the `REQUEST LOG`/`HEADERS`/
    `SAMPLING PARAMS`/`MODEL SETTINGS`/`REQUEST`/`PROMPT` sections in one
    call.
  - `write_chunk(text)` — appends one decoded response-text delta; opens the
    `=== RESPONSE (streaming) ===` header on first call only.
  - `write_full_response(text)` — non-incremental fallback (used only when
    truly nothing else was possible — see the DFlash caveat in Part 3's
    "gotchas" section for why this fallback exists at all).
  - `write_footer(...)` — the performance block; accepts a `label` kwarg so
    the identical formatting code produces both `=== PERFORMANCE ===` and
    `=== PERFORMANCE (partial) ===`.
  - `write_error(exc, *, partial=None)` — the error block, then calls
    `write_footer(label="PERFORMANCE (partial)", **partial)` if partial data
    was given.
  - `close()` — idempotent; safe to call from multiple except-handlers on
    the same writer without double-closing issues.
- `NullRequestLogWriter` — identical interface, every method a no-op. The
  factory function `get_request_log_writer(global_settings, ...)` returns
  this when the setting is off, so call sites in `server.py` never branch on
  the setting themselves.

### Scheduler instrumentation (`omlx/scheduler.py`, `omlx/request.py`)

Added three fields to the internal per-request state object, threaded
through to the API layer via the output objects:

- `prefill_started_at` — set once, at the single point a request leaves the
  waiting queue and is admitted (`Scheduler._schedule_waiting`, right after
  `self.waiting.popleft()`). `prefill_started_at - arrival_time` = pure
  queue-wait.
- `concurrent_at_prefill_start` / `concurrent_at_decode_start` — snapshot of
  "how many other requests were already admitted" at each transition, via a
  small helper (`Scheduler._concurrent_others_excluding(request)`) that
  checks both `self.running` and `self.prefilling` since the request may or
  may not already be counted in either depending on the exact call site
  (chunked-prefill continuation vs. direct admission both call the same
  insertion codepath from different states — don't assume a fixed offset,
  check membership explicitly).

These fields are set on the internal request object, then copied onto the
per-step output object at the point it's actually built (the existing
per-decode-step output constructor), and then again copied through every
engine-layer wrapper that turns the internal output into the
API-layer-visible output type. **Every wrapping layer between "the
scheduler knows this" and "the HTTP handler can read this" needs the field
added explicitly** — there's no shortcut; grep for every place the
lower-level output type gets wrapped into the higher-level one and update
all of them, or the field silently stays `None`/`0` for some code paths.

### `server.py` integration

Two endpoints instrumented: `/v1/completions` and `/v1/chat/completions`
(both streaming and non-streaming variants), plus a third internal branch
(markdown-extraction requests) that shares the chat endpoint's URL but
dispatches to separate handler functions.

**Non-streaming builders now consume the streaming engine API internally**
(`_build_completion`/`_build_chat_completion` call
`engine.stream_generate()`/`engine.stream_chat()` instead of
`engine.generate()`/`engine.chat()`), aggregating chunks themselves and
writing each one to the log via `write_chunk()` as it arrives, then using
the final chunk's cumulative text/token-counts to build the same JSON
response as before. This was the single biggest behavior change and the one
most likely to have hidden gotchas — see the DFlash caveat below.

**Every generation loop tracks two variables, not one:**

```python
last_output = None            # most recent chunk, for general logic
last_output_with_tokens = None  # most recent chunk that actually had token counts

async for chunk in engine.stream_x(...):
    ...
    last_output = chunk
    if getattr(chunk, "completion_tokens", 0) or getattr(chunk, "prompt_tokens", 0):
        last_output_with_tokens = chunk
```

and the partial-footer / concurrency-field lookups all prefer
`last_output_with_tokens or last_output` — this is the fix for gotcha #2
above (zero-token abort marker overwriting real data).

**Every generation loop catches cancellation explicitly:**

```python
except (Exception, asyncio.CancelledError) as exc:
    ...
    log_writer.write_error(exc, partial={...})
    log_writer.close()
    raise   # always re-raise — never swallow a cancellation
```

`asyncio.CancelledError` is a `BaseException` in Python ≥3.8, not an
`Exception` — a bare `except Exception` silently misses it. This is gotcha
#1 above.

**One engine-specific gotcha (DFlash speculative-decoding engine):** its
streaming code path accumulates cumulative response text from raw
per-chunk deltas, while its non-streaming path accumulates from
parser-finalized text (thinking/tool-call markup stripped) — the two are
*not* interchangeable whenever an output parser (Harmony, Gemma4-style tool
calling, etc.) is active. A capability check
(`engine.has_active_output_parser`) gates whether the "stream internally"
optimization is safe to apply per-request; when unsafe, that one request
falls back to the old atomic call (no incremental log visibility for that
specific combination, matching pre-feature behavior — not a regression, just
not a universal win). **Before assuming your backend's streaming and
non-streaming code paths produce identical final text, verify it** —
especially if there's any post-processing (tool-call extraction, thinking-tag
stripping, output-format parsers) that might be applied in one path and not
the other.

### Testing

- `tests/test_request_logging.py` — pure unit tests for the module
  (filename generation, writer output content, footer field
  presence/omission), no server/engine involved.
- `tests/test_request_logging_cancellation.py` — directly drives the
  streaming generator functions with a fake engine that yields one chunk
  then hangs, throws `CancelledError` into the generator mid-stream, and
  asserts the log writer got an error + partial footer with real
  (non-zero) token counts, then re-raised.
- `tests/integration/test_server_endpoints.py` (`TestNonStreamingRequestLogging`)
  — full `TestClient`-based HTTP tests with a fake engine implementing both
  the atomic and streaming methods, verifying: (a) a non-streaming HTTP
  request still produces a `=== RESPONSE (streaming) ===` section (proving
  the internal-streaming consumption actually ran), (b) a fake engine whose
  streaming method raises partway through leaves accurate partial token
  counts in the log — this is the regression test for the exact production
  incident that motivated the `last_output_with_tokens` fix.

---

## Part 3 — CachyLlama porting plan

Target: `~/git/dl/CachyLLama` (llama.cpp fork, C++, cpp-httplib server under
`tools/server/`).

> **Status**: implemented on `feature/request-logging`
> (`git@github.com:mikael-johansson/CachyLLama.git`), predating three
> readability refinements made to the oMLX reference implementation after
> initial port: the performance-footer table format, the leading blank line
> before the header marker, and always showing the opening `<think>` tag in
> the response log (see Part 1's "File format" and "Reasoning-model
> `<think>` prefix" sections — those are now the canonical format; the rest
> of this Part 3 section, written before the port, is historical context for
> *why* things are structured the way they are, not a literal diff against
> current `feature/request-logging`).

### The good news: no dual-path problem

Unlike oMLX's Python engine (which had genuinely separate `generate()` /
`stream_generate()` implementations), CachyLlama's server **already
generates token-by-token internally regardless of HTTP streaming mode.**
`server_routes::handle_completions_impl(...)`
(`tools/server/server-context.cpp:4913`) always builds `server_task`s and
posts them to a task queue consumed by the slot loop one token at a time.
The only branch between streaming and non-streaming is *how the HTTP
handler reads the results* (`server-context.cpp:5043-5108`):

- non-streaming: `rd.wait_for_all(req.should_stop)` — blocks and collects
  every chunk into `all_results` before replying once
- streaming: `rd.next(req.should_stop)` per SSE frame

**This means gotcha #3 from Part 1 (force streaming internally) is free
here** — you don't need to change how generation happens, you only need to
tap the existing per-chunk result stream in one place that both the
streaming and non-streaming reader paths already go through, or hook both
`rd.next()` and `rd.wait_for_all()` call sites identically.

### Existing precedent to build on

Two things already exist in this codebase that are directly relevant —
extend rather than reinvent:

1. **`params.path_prompts_log_dir`** (`common/common.h:507`) — an existing
   *prompt-only* disk logger, already wired to a CLI flag
   `--log-prompts-dir` (`common/arg.cpp:3909-3919`) and used in
   `handle_completions_impl` (`server-context.cpp:4937-4945`), writing each
   prompt to `<dir>/%012lld.txt`. This is the closest existing analog to
   what you're building — same shape (opt-in directory, one file per
   request), just needs to grow into full request+response+performance
   logging and a smarter filename (see Part 1's naming scheme vs. this
   existing numeric one).
2. **`log_server_request(...)`** (`tools/server/server-http.cpp:30-48`) — a
   full HTTP request/response body logger that already exists but is
   **disabled** (`server-http.cpp:127`: `// srv->set_logger(log_server_request); // TODO ... this is too spammy`).
   This is essentially gotcha-free header/body access already solved; the
   reason it's off is exactly why the real feature needs to be opt-in and
   per-request-file rather than one giant interleaved debug log. Worth
   reading for the header/body access pattern even though the new feature
   should not just be "turn this on."

### Data mapping (oMLX concept → CachyLlama equivalent)

| Part 1 concept | CachyLlama field/mechanism | Location |
|---|---|---|
| in-flight request | `server_task` | `tools/server/server-task.h:136-277` |
| per-request runtime/timing state | `server_slot` | `tools/server/server-context.cpp:164+` |
| `arrival_time` | *(not currently tracked — see "new field needed" below)* | — |
| `prefill_started_at` | `slot.t_start_process_prompt` | `server-context.cpp:284-290` |
| `generation_started_at` | `slot.t_start_generation` | `server-context.cpp:284-290` |
| prefill duration | `slot.t_prompt_processing` | `server-context.cpp:284-290` |
| generation duration | `slot.t_token_generation` | `server-context.cpp:284-290` |
| `cached_tokens` (prefix reuse) | `slot.n_prompt_tokens_cache` | `server-context.cpp:199`, computed at `:3856-3858` |
| `prompt_tokens` processed this request | `slot.n_prompt_tokens_processed` | `server-context.cpp:200` |
| completion tokens | `slot.n_decoded` | `server-context.cpp:193-196` |
| per-request final result object | `server_task_result_cmpl_final` | `tools/server/server-task.h:357` |
| per-request streaming chunk object | `server_task_result_cmpl_partial` | `tools/server/server-task.h:432` |
| existing "timings" summary (already close to the footer!) | `result_timings` struct | `tools/server/server-task.h:279-297` (`cache_n`, `prompt_n`/`prompt_ms`, `predicted_n`/`predicted_ms`) |
| HTTP headers | `server_http_req::headers` (`std::map<string,string>`) | `tools/server/server-http.h:51`, populated at `server-http.cpp:518` |
| client disconnect signal | `req.is_connection_closed` captured as `should_stop` | `server-http.cpp:590` (GET), `:637` (POST) |
| cancellation mechanism | cooperative: `server_response_reader::stop()` posts `SERVER_TASK_TYPE_CANCEL`; always called from the reader's RAII destructor | `tools/server/server-queue.cpp:441-459`, `server-queue.h:191-193` |
| mid-generation abort (OOM/context-exceeded/decode failure) | `abort_all_slots(reason)`, called from `catch` blocks around decode stages | `server-context.cpp:3238-3245`, call sites at `:3319-3321`, `:3370-3372`, `:3379-3381` |
| context-length-exceeded (specific abort) | `ERROR_TYPE_EXCEED_CONTEXT_SIZE` path | `server-context.cpp:3835-3854` |
| concurrency count ("how many other requests active") | **no thread-safe getter exists yet** — see below | — |
| logging macros | `SRV_*`/`SLT_*` wrapping `LOG_*` | `tools/server/server-common.h:24-36` |

### `arrival_time` / queue-wait: needs a new field

There's no existing "task was enqueued at time X, distinct from when
processing started" timestamp visible in the investigation. If queue-wait
visibility matters for CachyLlama's use case (it's a bigger concern in
multi-tenant/concurrent-heavy deployments than single-user local serving),
add a timestamp to `server_task` (`tools/server/server-task.h`) set at task
construction/`post_tasks()` time, mirroring how oMLX added
`prefill_started_at` to its `Request` — a small, additive field, not a
structural change. If it's not worth the churn for this deployment, just
omit `queue_wait_sec` from the footer (see Part 1: omit, don't fake).

### Concurrency count: needs a new mechanism, use the existing pattern

`slots` (`std::vector<server_slot>`, `server-context.cpp:927`) is only
safely touched by the queue/model thread. There's no direct getter callable
from an HTTP handler thread today — the only existing way to learn "how
many slots are busy" is posting a `SERVER_TASK_TYPE_METRICS` task and
awaiting the async result (what `/metrics` and `/slots` do,
`server-context.cpp:2984-3004`, `:5354-5387`), which is too heavyweight to
do on every request's log-write path.

Follow the existing **`get_active_user_count()`** pattern instead
(`server-context.cpp:869-875`) — it already takes `queue_tasks.mutex_tasks`
and reads a small map synchronously from a handler thread. Add a similarly
small atomic counter or mutex-guarded integer, incremented/decremented at
the same places slots transition to/from "processing" (near
`metrics.on_decoded()`, `server-context.cpp:831-835`, which already tracks
a *cumulative* busy-slot counter — you want an instantaneous one alongside
it). Snapshot it once when a request's prefill starts and once when decode
starts, same as oMLX's `concurrent_at_prefill_start` /
`concurrent_at_decode_start`.

### Cancellation handling: RAII does most of the work for you

This is architecturally *cleaner* than oMLX's Python `CancelledError`
handling, not harder:

- `server_response_reader::stop()` (`server-queue.cpp:441-459`) is called
  automatically from the reader's destructor
  (`server-queue.h:191-193`) — meaning **any early return, exception, or
  handler-scope exit already triggers cancellation cleanup**, no manual
  `except`/`catch`-and-rethrow needed the way Python required.
- The place to hook is: wherever you detect `is_terminated` /
  `should_stop()` returned true (mirroring `server-context.cpp:5048-5049`
  for the non-streaming path, and wherever the streaming SSE callback checks
  the equivalent condition) — write the `=== ERROR ===` (or a distinct
  `=== CANCELLED ===` label, since this isn't really an "exception" in C++
  terms) block plus partial footer there, using the current slot's
  `n_decoded`/`generated_text`/timing fields **before** `slot.release()`
  runs (verify the exact ordering when implementing — see "must verify"
  below).

### Must verify before shipping (the C++ analog of oMLX's token-clobbering bug)

oMLX's real production bug (partial token counts silently zeroed by a
final abort marker) came from an ordering assumption that turned out to be
wrong. Don't assume CachyLlama is immune — **explicitly test** this
scenario before considering the port done:

1. Start a request with a small `n_predict`/`max_tokens` but force an abort
   partway (e.g. via `abort_all_slots` triggered by a context-shift failure,
   or by killing/reducing available memory) after some real tokens have
   been generated.
2. Confirm that whatever data your logging hook reads at that point (slot
   timing fields, `n_decoded`, generated text so far) reflects the *real*
   partial progress, not zeros — i.e. confirm `send_error(slot, reason, ...)`
   (`server-context.cpp:3238-3245` and call sites) is invoked **before**
   anything that would reset `slot.n_decoded`/`generated_text`/timing
   fields, and that `slot.release()` (called right after `send_error`) is
   the point after which those fields are no longer trustworthy — read them
   before that line, not after.
3. Write this as an automated test (see below) so it can't silently regress.

### Config flags

Follow the `--log-prompts-dir` pattern exactly
(`common/arg.cpp:3909-3919`, fields in `common/common.h` near `:507`):

```cpp
// common/common.h, near path_prompts_log_dir
bool  request_logging_enabled = false;
std::string path_request_log_dir; // empty = disabled (or reuse the enabled bool + a required dir)
```

```cpp
// common/arg.cpp, near the --log-prompts-dir registration
add_opt(common_arg(
    {"--request-logging-dir"}, "PATH",
    "Log full request/response transcripts to directory, one file per "
    "request (auto-created if not present; default: disabled)",
    [](common_params & params, const std::string & value) {
        params.path_request_log_dir = value;
        params.request_logging_enabled = true;
        std::error_code ec;
        std::filesystem::create_directories(value, ec);
        ...
    }
));
```

Matches Part 1's "opt-in, off by default" decision. Decide independently
whether header redaction / retention pruning should differ from the oMLX
defaults for this deployment (Part 1 flags these as decisions to
re-confirm, not copy blindly).

### Suggested implementation order

1. Add the config flag(s) (mechanical, low risk, following the existing
   `--log-prompts-dir` pattern exactly).
2. Write the filename/formatting logic as a small standalone header+source
   pair (e.g. `tools/server/server-request-log.h/.cpp`), mirroring
   `omlx/request_logging.py`'s structure: a path-builder function, a writer
   class with `write_header`/`write_chunk`/`write_footer`/`write_error`, a
   null-writer for the disabled case. Unit-testable in isolation if the
   build supports a plain C++ test binary for `tools/server/`; otherwise
   test it through the Python integration harness (see below).
3. Hook `write_header` once per request in `handle_completions_impl`
   (`server-context.cpp:4913`), right after the request is parsed/validated
   and before task posting — headers, model, sampling params, raw body, and
   the readable prompt transcript are all available at that point.
4. Hook `write_chunk` at the point both `rd.next()` (streaming) and
   `rd.wait_for_all()` (non-streaming)'s internal per-chunk collection loop
   consume a `server_task_result_cmpl_partial`/`_final` — ideally one shared
   spot both paths flow through, so you don't duplicate the hook (re-check
   the exact call graph at `server-context.cpp:5043-5108` when
   implementing; the investigation found the branch point but not
   necessarily a single shared per-chunk callback — if there isn't one,
   hook both `rd.next()`'s streaming-callback registration and
   `wait_for_all()`'s internal collection loop identically).
5. Hook `write_footer` (success) / `write_error` + partial footer
   (abort/cancel/context-exceeded) at the existing `send_error(...)` call
   sites (`server-context.cpp:3238-3245` and friends) and at the
   cancellation-detection point identified above — reading slot fields
   before `slot.release()`, per the "must verify" section.
6. Add the concurrency counter (step can be deferred/omitted per Part 1's
   "omit uncomputable fields" — footer just won't have
   `concurrent_requests_at_*` lines until this exists).
7. Add `arrival_time`/queue-wait tracking if wanted (also deferrable).
8. Tests (see below), including the abort-ordering verification.

### Testing

CachyLlama already has a real integration test harness for the server:
`tools/server/tests/` — pytest-based, spins up the actual server binary and
drives it over HTTP (`tools/server/tests/unit/test_*.py`, e.g.
`test_stream.py`, `test_completion.py`, `test_ctx_shift.py`). Add
`tools/server/tests/unit/test_request_logging.py` following that pattern:

- Start the server with `--request-logging-dir <tmp>`.
- Fire a normal completion, assert a log file appears with the expected
  section markers and plausible field values.
- Fire a streaming completion, assert the response section is populated
  incrementally (or at least present) and matches what was streamed to the
  client.
- Force a context-length-exceeded or similar abort mid-generation (there's
  already `test_ctx_shift.py` to crib the setup from) and assert the
  resulting log file has an error block **and** non-zero partial token
  counts — this is the regression test for the "must verify" ordering
  concern above, and the single most important test in the whole port,
  since it's the exact bug class that motivated this entire feature.
- Confirm no file is created when the flag is off (default).

---

## Part 4 — vLLM porting plan (this repo, `qwen38-27b-rtx3090`, single-user mode)

Target: this repo, which does not vendor a vLLM fork — it installs stock vLLM
0.27.1 into `venv/` and applies small `patch -p1` diffs against the installed
package (`patches/*.patch`, verified by `verify.sh`). This plan follows that
existing convention rather than forking vLLM. All file:line references below
are against `venv/lib/python3.12/site-packages/vllm` at 0.27.1, verified
directly against the installed tree, not recalled from general vLLM
knowledge — re-check them at implementation time regardless, per this repo's
own gotcha #5 (compiled-cache/version drift) and the general lesson of this
document (**Part 3 belongs to a different codebase; treat even this Part 4
as needing re-verification once written, the same way Part 3 says of
itself**).

Server entrypoint: `vllm serve <model> ...` (CLI, OpenAI-compatible HTTP
server, FastAPI/Starlette + Uvicorn), V1 engine, `AsyncLLM`,
`--api-server-count 1` in both `single-user/start_qwen.sh` and
`batch/start_qwen.sh` (this plan's concurrency-count approach assumes 1 API
process; see "Concurrency count" below for what breaks above that).

### The key difference from CachyLlama: a real dual-path problem, with a clean fix

CachyLlama got gotcha #3 (route non-streaming through the incremental path)
for free, because its server already generates token-by-token internally
regardless of HTTP mode. **vLLM does not.** `ChatCompletionRequest.to_sampling_params`
(`entrypoints/openai/chat_completion/protocol.py:722-724`) sets:

```python
output_kind=(
    RequestOutputKind.DELTA if self.stream else RequestOutputKind.FINAL_ONLY
)
```

(`completion/protocol.py:375-377` is identical for the legacy endpoint.) With
`FINAL_ONLY`, the V1 output processor **never emits an intermediate
`RequestOutput` at all** — `v1/engine/output_processor.py:288-289`:
`if not finished and final_only: return None`. So for a non-streaming
request today, `chat_completion_full_generator`
(`chat_completion/serving.py:844`) only ever sees one item:

```python
async for res in result_generator:      # serving.py:860
    final_res = res
except asyncio.CancelledError:          # serving.py:862-863
    return self.create_error_response("Client disconnected")
```

If a non-streaming request hangs, gets cancelled, or aborts mid-generation,
there is currently **no partial `RequestOutput` to read at all** — not just
no incremental log, literally nothing, because the engine only ever
constructs one at the very end. This is a strictly harder starting point
than CachyLlama's llama.cpp server (which always has *some* per-token state
to read, even mid-abort) or oMLX's Python engine (which had a real streaming
method to fall back to). Naively "just tap `result_generator` in both
generator methods" is insufficient here — for non-streaming, there is
nothing to tap until the request has already finished or died.

**The fix**, and the one clean architectural insight this port needs: there
is a third `RequestOutputKind`, `CUMULATIVE`
(`sampling_params.py:182-188`) — *"Return entire output so far in every
`RequestOutput`"* — as opposed to `DELTA`'s "only the delta." If, only when
request-logging is enabled **and** the request is non-streaming, the patch
overrides `sampling_params.output_kind = RequestOutputKind.CUMULATIVE`
right after it's built (`chat_completion/serving.py:304`,
`completion/serving.py:178`), two things follow:

1. The output processor now emits an intermediate `RequestOutput` on every
   engine step (same `if not finished and final_only: return None` check no
   longer short-circuits, since `final_only` is now false) — solving the
   "nothing to tap" problem, tappable exactly like the streaming path.
2. **`chat_completion_full_generator`'s existing `final_res = res` loop
   needs zero changes.** Under `CUMULATIVE`, each yielded output's
   `.text`/`.token_ids` already *is* the complete text so far, not a delta —
   so the last iteration's `res` is still the complete, correct response by
   construction. Unlike oMLX (which had to rewrite non-streaming builders to
   aggregate deltas themselves) and unlike a naive "just force `DELTA`"
   approach (which would silently corrupt the final response into just the
   last chunk's delta, since nothing in `full_generator` accumulates), this
   is a two-line change to which enum value gets set, with the existing
   aggregation logic staying correct for free. **This must be verified**
   once implemented — confirm `final_res.outputs[i].text` after the loop
   really is the full response, not a delta, before trusting it; this is
   this port's version of Part 3's "must verify before shipping" ordering
   check.

Do **not** set `CUMULATIVE` (or touch `output_kind` at all) for streaming
requests — the streaming generator's own accumulation
(`previous_texts[i] += delta_text`, `chat_completion/serving.py:629`) assumes
`DELTA` and is part of the actual OpenAI SSE contract the client depends on;
changing it would corrupt real client-facing output, not just the log.

### Existing precedent to build on

1. **`patches/*.patch` + `verify.sh`** — this repo's own established
   mechanism for modifying installed vLLM source, already applied to 7
   other features (README "Why this isn't just `vllm serve`"). `verify.sh`
   already loops over every file in `patches/` and checks it's applied
   (`for p in patches/*.patch; do patch -p1 -R --dry-run ...`); dropping a
   new `patches/request-logging.patch` in that directory gets verification
   for free with no changes to `verify.sh` itself.
2. **`vllm.entrypoints.serve.utils.request_logger.RequestLogger`**
   (`entrypoints/serve/utils/request_logger.py`) — vLLM's own built-in
   request logging (`--enable-log-requests` /`--enable-log-outputs`). It
   only calls Python's `logging` module (one line per request/output to the
   process log, not files), so it's not reusable as the writer itself, but
   its call sites are the exact right hook points to mirror: `log_inputs()`
   is called from `GenerateBaseServing._log_inputs`
   (`entrypoints/serve/engine/serving.py:95-111`), itself called from
   `chat_completion/serving.py:309` and `completion/serving.py:185` — i.e.
   right after the request is validated/rendered, same place this plan
   writes the header.
3. **`--middleware <dotted.path>`** (`entrypoints/openai/cli_args.py:301-306,344-348`,
   wired in `api_server.py:339-347` via plain `importlib`) — a supported,
   no-source-patch way to add ASGI middleware. **Considered and rejected as
   the primary mechanism**: middleware sees raw HTTP send/receive events,
   not `RequestOutput`/`SamplingParams`/parsed messages — the per-chunk
   token/metrics/reasoning-split logging this feature needs happens inside
   `OpenAIServingChat`/`OpenAIServingCompletion` methods, which middleware
   cannot reach without monkeypatching those classes at import time anyway
   (no cleaner than a source patch, and more upgrade-fragile: a patch fails
   loudly with a hunk-rejection, a monkeypatch on a renamed/reshaped method
   fails silently as a no-op). Use the existing `patches/` convention
   instead, consistent with how this repo already treats vLLM as a
   patched vendor dependency.

### Data mapping (oMLX/CachyLlama concept → vLLM 0.27.1 field/mechanism)

| Part 1 concept | vLLM field/mechanism | Location |
|---|---|---|
| in-flight request | `RequestOutput` (per chunk) | `outputs.py` |
| per-request timing state | `RequestStateStats` (`arrival_time`, `queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`, `num_generation_tokens`) | `v1/metrics/stats.py:218-236`, attached as `RequestOutput.metrics` at `v1/engine/output_processor.py:385` |
| `arrival_time` | `RequestStateStats.arrival_time` — **already tracked, no new instrumentation needed** (unlike CachyLlama, which had to add `t_arrival_us`) | same |
| queue-wait | `scheduled_ts - queued_ts` (or `- arrival_time`) on `RequestOutput.metrics` | same |
| TTFT | `first_token_ts - arrival_time` (or vLLM's own derivation, see below) | same |
| prefill/generation duration | derivable from `scheduled_ts`/`first_token_ts`/`last_token_ts` | same |
| already-derived footer fields | `build_per_request_timing_metrics()` → `PerRequestTimingMetrics` (`ttft_ms`, `generation_time_ms`, `queue_time_ms`, `mean_itl_ms`, `tokens_per_second`) | `entrypoints/generate/base/serving.py:46-96`, requires `log_stats` (default on: `disable_log_stats: bool = False`, `engine/arg_utils.py:540`) — **prefer computing footer fields directly from `RequestOutput.metrics` per chunk** rather than this helper, since the helper is only invoked once per request (`serving.py:1082` non-streaming, `:783-792` streaming) whereas the log writer needs it incrementally |
| `cached_tokens` (prefix reuse) | `RequestOutput.num_cached_tokens` | `outputs.py:105,124,147` — present on **every** chunk, not just first; chat serving.py reads it only on `first_iteration` (`:493`) since it's stable after prefill, but nothing stops reading it every chunk |
| cumulative vs delta text | `CompletionOutput.text`/`.token_ids` — delta under `RequestOutputKind.DELTA` (streaming default), cumulative under `CUMULATIVE` (see above), single-and-complete under `FINAL_ONLY` (vanilla non-streaming, before this patch) | `outputs.py:22-48`, enum at `sampling_params.py:182-188` |
| count of other requests running | `self.engine_client.output_processor.get_num_unfinished_requests()` — plain `len(dict)`, synchronous, no round-trip, `AsyncLLM`-specific (not on the abstract `EngineClient`) | `v1/engine/output_processor.py:449-450`, `AsyncLLM.output_processor` at `v1/engine/async_llm.py:141` — **only valid in-process with `--api-server-count 1`**; with more API server processes each has its own `AsyncLLM`/`output_processor` and would only see its own share, so this field should be omitted (not faked) if this ever runs with `API_SERVERS>1` |
| HTTP headers / client addr / raw body | `raw_request: Request` (Starlette) — already threaded into `create_chat_completion`/`_create_chat_completion` (`chat_completion/serving.py:219-233,235-`) and `create_completion` (`completion/serving.py:113-`); `raw_request.headers`, `raw_request.client.host`, `await raw_request.body()` (safe to call again post-validation — Starlette caches it) | same |
| cancellation mechanism | `@with_cancellation` decorator on the route function (`entrypoints/serve/utils/api_utils.py:52-94`) races the handler against `listen_for_disconnect()`, cancels the loser; **once a `StreamingResponse` is returned it stops listening and Starlette's own `StreamingResponse` cancellation takes over instead** | `api_utils.py`, applied at `chat_completion/api_router.py:51-53` |
| reasoning/`<think>` forced-open detection | `parser.is_reasoning_end(prompt_token_ids)` — already computed per-request for vLLM's own parsing | `chat_completion/serving.py:339` |
| raw (pre-reasoning-split) generated text | `output.text` — read at `chat_completion/serving.py:609` (streaming, before `parser.parse_delta(...)`) and `:900`-ish (non-streaming, before `parser.parse(...)`) | same |

### Config

A plain environment variable, `REQUEST_LOG_DIR` (unset = disabled — matches
Part 1's opt-in-by-default-off rule), read directly by the patched code via
`os.environ.get(...)`. Deliberately **not** registered in
`vllm/envs.py` the way `patches/speed-knobs-envs.patch` registers the speed
knobs — that registration exists because those knobs change tensor/Marlin
workspace shapes baked into the `torch.compile` graph (this repo's gotcha
#5); `REQUEST_LOG_DIR` never touches model computation, so it doesn't need
to be part of the compile cache key and skipping the `envs.py` patch keeps
this feature's diff smaller and independent of the other patches.

Wire it into `single-user/start_qwen.sh` (and `batch/start_qwen.sh` if
wanted there too — see "Concurrency count" caveat above before wiring
`batch/`) the same way `PORT`/`MAX_SEQS`/etc. are: `export REQUEST_LOG_DIR`
before the `exec venv/bin/vllm serve ...` line, and add a row to
`single-user/README.md`'s Knobs table.

### The writer module

A standalone `request_logging.py` at the repo root (alongside
`quant_lm_head.py`, `build_draft_vocab.py`, etc. — not inside the vendored
`vllm` package), mirroring `omlx/request_logging.py`'s shape:

- `build_request_log_path(requests_dir, *, prompt_text, request_id)` — same
  filename scheme as Part 1 (ISO8601 UTC timestamp, sanitized request-id
  capped ~40 chars, first-10-words prompt stub capped ~80 chars, `-2`/`-3`
  collision suffixes). Port directly from CachyLlama's actual
  `server_request_log_build_path` (`tools/server/server-request-log.cpp`,
  ~line 90-115 as of the `request-logging` branch) — same algorithm, trivial
  Python translation.
- `render_chat_transcript(messages)` — `SYSTEM:`/`USER:`/`ASSISTANT:`/`TOOL:`
  blocks, `[image attached]`/`[audio attached]`/`[video attached]`/
  `[attachment: type]` for non-text content parts, `[tool_call: name(args)]`
  for `tool_calls`. Port directly from CachyLlama's
  `server_request_log_render_chat_messages` — the input shape (OpenAI
  `messages` array) is identical since vLLM's `ChatCompletionRequest.messages`
  is the same wire format CachyLlama parses from `req.body`'s `"messages"`
  key.
- `RequestLogWriter` — one per request:
  - `write_header(request_id, client_addr, method, path, model,
    resolved_model, headers, sampling_params, model_settings, raw_body,
    prompt_text, thinking_forced_open)` — opens the file, writes
    `REQUEST LOG`/`HEADERS`/`SAMPLING PARAMS`/`MODEL SETTINGS`/`REQUEST`/
    `PROMPT` sections, in one call, matching Part 1's exact section order
    and the leading-blank-line / aligned-footer-table refinements already
    reflected in Part 1 (confirmed against the actual CachyLlama code, not
    just the spec text — see `git show e01d2d57c` and `42e1ac2f0` on the
    `request-logging` branch of `~/git/cachy-fork` for ground truth).
  - `write_chunk(text)` — appends one piece of raw response text; opens
    `=== RESPONSE (streaming) ===` on first call, writing the synthetic
    `<think>\n` opener first if `thinking_forced_open` was true.
  - `note_progress(prompt_tokens, completion_tokens, cached_tokens,
    metrics: RequestStateStats | None, concurrency_snapshot)` — updates
    tracked footer state; called once per chunk. Keep a `last_with_tokens`
    variable distinct from "most recent," per Part 1 gotcha #2, even though
    vLLM's cumulative/delta model looks less prone to an in-band
    zero-valued abort marker than CachyLlama's interleaved-result-stream
    design (errors surface as raised exceptions from iterating
    `result_generator`, not as a special zero-count item in the stream) —
    keep the defensive tracking anyway; it's cheap insurance against an
    edge case (e.g. a final chunk with an empty `CompletionOutput` list)
    that wasn't fully ruled out during this research pass, and must be
    exercised by the disconnect/cancellation test either way.
  - `write_footer(label="PERFORMANCE")` / `write_error(exc, http_status=None)`
    (calls `write_footer(label="PERFORMANCE (partial)")` internally) — same
    aligned-table renderer as Part 1/CachyLlama: label column left-justified
    to the longest **included** row, values always carry their unit
    (`tok/s`, `s`, `%`), omit uncomputable rows rather than writing `0`.
  - `close()` — idempotent.
- `NullRequestLogWriter` — every method a no-op.
- `get_request_log_writer(request_id, prompt_text)` factory — reads
  `REQUEST_LOG_DIR` from the environment itself, so call sites in the vLLM
  patch never branch on the setting (same discipline as oMLX/CachyLlama).

### Patch hook points (`patches/request-logging.patch`)

Both `chat_completion/serving.py` and `completion/serving.py` get the same
five hooks; line numbers are for the chat file (0.27.1), with the
completion-file equivalents noted from the same research pass — **re-verify
both against the actual installed tree at implementation time**, this
codebase's `entrypoints/openai/` layout has already changed across versions
once (older vLLM docs/blog posts refer to a flat `serving_chat.py` that no
longer exists in 0.27.1).

1. **Import + header write**, in `_create_chat_completion`
   (`chat_completion/serving.py:235`), after `render_chat_request` resolves
   `conversation`/`engine_inputs` (~line 253) and after `sampling_params =
   request.to_sampling_params(...)` (line 304), before `generator =
   self.engine_client.generate(...)` (line 343): build the writer via
   `get_request_log_writer(request_id, prompt_text)`, render
   `prompt_text` from `conversation` via `render_chat_transcript`, compute
   `thinking_forced_open` from `parser.is_reasoning_end(prompt_token_ids)`
   negated (parser's own check, already computed nearby at line 339 — reuse
   it, don't re-derive), call `write_header(...)`. Store the writer on
   something reachable by both generator methods (e.g. a local passed
   through, or `raw_request.state.request_log_writer` alongside the
   existing `raw_request.state.request_metadata` pattern at line 264).
   Completion-file equivalent: `create_completion`
   (`completion/serving.py:113`), sampling params at `:178`, generate() at
   `:207` — prompt text here is the resolved prompt string directly (no
   `messages` to render), matching Part 1's fallback-to-resolved-prompt
   case.
2. **`CUMULATIVE` override for logged non-streaming requests**, right after
   step 1's `sampling_params` is built: `if writer is not None and not
   request.stream: sampling_params.output_kind =
   RequestOutputKind.CUMULATIVE` (needs `RequestOutputKind` added to the
   existing `from vllm.sampling_params import BeamSearchParams,
   SamplingParams` line, `chat_completion/serving.py:64`).
3. **Streaming chunk tap**, in `chat_completion_stream_generator`
   (line 422), inside the `async for res in result_generator:` loop
   (line 485): call `writer.write_chunk(output.text)` where `delta_text`
   is computed (line 609) — log the **raw** delta before
   `parser.parse_delta(...)` is applied, per Part 1's "log is not a
   protocol payload" reasoning about the `<think>` tag; call
   `writer.note_progress(...)` using `res.metrics`/`res.num_cached_tokens`
   each iteration. Wrap the loop body in `try`/`finally` (not just
   `except Exception`) so a client disconnect — which, per the cancellation
   row above, is delivered as a generator cancel *after* `with_cancellation`
   has handed off to `StreamingResponse`, not as a caught exception in this
   method today — still runs `writer.write_error(...)`/`close()`. This is
   this port's version of Part 1 gotcha #1 (`CancelledError` is a
   `BaseException`): today's `chat_completion_stream_generator` has **no**
   `except asyncio.CancelledError` at all (only `except GenerationError`/
   `except Exception` later in the method) — confirm this during
   implementation and add the `finally` regardless of what's already there,
   don't assume.
4. **Non-streaming chunk tap**, in `chat_completion_full_generator`
   (line 844), inside `async for res in result_generator: final_res = res`
   (line 860): same `write_chunk`/`note_progress` calls, computing the
   delta to log as `res.outputs[i].text` minus what's already been written
   (since `CUMULATIVE` means each `res` carries the full text so far —
   track a per-choice "already logged up to N chars" offset, don't
   re-log the whole cumulative string every chunk). The existing `except
   asyncio.CancelledError: return self.create_error_response(...)`
   (line 862-863) already exists here — just add
   `writer.write_error(...)` inside it.
5. **Concurrency snapshot**: once at header-write time (step 1, "count at
   arrival") and once on the first chunk received in each tap (step 3/4,
   approximating CachyLlama's `concurrent_at_prefill_start`/
   `concurrent_at_decode_start` — vLLM doesn't expose a clean
   prefill-start-vs-decode-start boundary to the serving layer the way
   CachyLlama's `server_slot` state machine does, so this is a **documented
   approximation**, not the precise two-point measurement Part 1 describes;
   say so in the footer field name or a comment, don't imply more precision
   than exists).

### Testing

- Pure unit tests for `request_logging.py` (filename generation, footer
  field presence/omission, transcript rendering) — no server, mirrors
  `omlx`'s `tests/test_request_logging.py` and CachyLlama's
  `tools/server/tests/unit/test_request_logging.py` structure.
- Manual integration checks against the running single-user server (this
  repo doesn't have a pytest-based server integration harness the way
  CachyLlama's `tools/server/tests/` does, so lean on `bash verify.sh` and
  hand-driven `curl`):
  - `REQUEST_LOG_DIR` unset → confirm no directory/files ever appear.
  - Streaming chat completion → log file appears, `RESPONSE (streaming)`
    section content matches exactly what the client received over SSE.
  - Non-streaming chat completion (this repo's own README curl example
    doesn't set `"stream": true`, so this is the default-documented path,
    not an edge case) → confirm a `RESPONSE (streaming)` section is
    present and incremental (this is the proof the `CUMULATIVE` override
    actually worked, same significance as CachyLlama's equivalent test)
    and that `final_res` in the actual HTTP JSON response still matches
    what's in the log — i.e. confirm step 2 of the architecture section
    above didn't silently corrupt the real response.
  - Client disconnect mid-generation (`curl --max-time` cutting a streaming
    request short, or closing the connection from a script) → confirm the
    log shows an error/cancelled block **and** non-zero partial
    `completion_tokens` — the single most important test in this port too,
    same bug class as both prior ports' most important test.
  - A request with `enable_thinking` on (this model's default) → confirm
    the log's `RESPONSE` section opens with `<think>` before the model's
    own reasoning text.
- Add `REQUEST_LOG_DIR` to `bash verify.sh`'s output as a WARN-if-unset info
  line (optional, low value) — the real verification vLLM patches get is
  already automatic via the existing `for p in patches/*.patch` loop once
  `patches/request-logging.patch` exists.

### Suggested implementation order

1. Write `request_logging.py` standalone (filename builder, writer class,
   null writer, transcript renderer) — testable without touching vLLM at
   all.
2. Unit tests for it.
3. `patches/request-logging.patch`, hook 1 (header write) + hook 2
   (`CUMULATIVE` override) only — get a log file with header sections
   appearing for both streaming and non-streaming requests, response
   section still empty/unwired.
4. Hook 3 (streaming tap) + hook 4 (non-streaming tap) — verify the
   non-streaming `CUMULATIVE` change didn't break the actual HTTP response
   (the single riskiest step in this whole plan) before moving on.
5. Hook 5 (concurrency snapshots) + footer/error wiring.
6. Cancellation `finally` block (streaming) — the gotcha #1 fix; test with
   a deliberately cut-short `curl`.
7. Reasoning `<think>` opener detection.
8. Wire `REQUEST_LOG_DIR` into `single-user/start_qwen.sh` +
   `single-user/README.md`'s Knobs table.
9. Manual test pass per "Testing" above, focused especially on the
   disconnect case and the non-streaming-response-correctness check.

### Open decisions to confirm before starting (per Part 1: these are product decisions, not universal truths)

- **Header redaction**: Part 1's oMLX default is verbatim, including
  `Authorization`. This repo's `api_key.txt` is a locally-generated bearer
  token for a single-user home deployment — verbatim logging seems fine
  under the same reasoning oMLX used, but confirm before shipping rather
  than inheriting it silently.
- **`batch/` mode**: this plan is scoped to `single-user/` as asked. If
  extended to `batch/` later, re-check the concurrency-count row above —
  `batch/start_qwen.sh` also defaults `API_SERVERS=1` today so the same
  approach would work as-is, but that's a fact to verify at the time, not
  assume holds forever.
- **Retention**: Part 1 default is no auto-pruning, operator manages disk
  manually — same recommendation here, no changes needed to adopt it.
