"""Unit tests for request_logging.py — no server/engine involved.

  python test_request_logging.py

Plain assert-based, matching this repo's other standalone test scripts
(marlin-tune/test_correctness.py, bench/test_spec_decode_attn.py) rather
than pytest, which isn't a dependency of this repo's venv.
"""

import tempfile
import types
from pathlib import Path

import request_logging as rl


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        return True
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")
        return False


def test_sanitize():
    assert rl._sanitize("hello world!!", 40) == "hello_world"
    assert rl._sanitize("a/b\\c:d", 40) == "a_b_c_d"
    assert rl._sanitize("", 40) == ""
    assert rl._sanitize("ab", 1) == "a"


def test_first_n_words_stub():
    assert rl._first_n_words_stub("Hello, how are you today my friend indeed") == (
        "Hello_how_are_you_today_my_friend_indeed"
    )
    assert rl._first_n_words_stub("") == "prompt"
    assert rl._first_n_words_stub("!!! ??? ...") == "prompt"
    long_word = "x" * 200
    stub = rl._first_n_words_stub(long_word)
    assert len(stub) <= 80


def test_build_request_log_path_basic_and_collision():
    with tempfile.TemporaryDirectory() as d:
        p1 = rl.build_request_log_path(d, prompt_text="Hello there world", request_id="req-1")
        p1.write_text("x")
        assert p1.name.endswith("_req-1_Hello_there_world.log")

        # Same second, same id/prompt -> collision suffix, never overwritten.
        p2 = rl.build_request_log_path(d, prompt_text="Hello there world", request_id="req-1")
        assert p2 != p1
        assert p2.name.endswith("-2.log") or p2.name != p1.name


def test_build_request_log_path_sanitizes_untrusted_id():
    with tempfile.TemporaryDirectory() as d:
        p = rl.build_request_log_path(d, prompt_text="hi", request_id="../../etc/passwd")
        assert "/" not in p.name
        assert p.parent == Path(d)


def test_render_chat_transcript():
    messages = [
        {"role": "system", "content": "Be nice."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        },
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"function": {"name": "search", "arguments": '{"q": "x"}'}}],
        },
    ]
    out = rl.render_chat_transcript(messages)
    assert "SYSTEM:\nBe nice." in out
    assert "USER:\nlook at this\n[image attached]" in out
    assert "[tool_call: search({\"q\": \"x\"})]" in out


def test_null_writer_is_all_noop():
    w = rl.NullRequestLogWriter()
    w.write_header(request_id="x")
    w.write_chunk("hello")
    w.note_progress(prompt_tokens=1, completion_tokens=1, cached_tokens=0, metrics=None)
    w.write_footer()
    w.write_error(ValueError("boom"))
    w.close()  # must not raise


def _fake_metrics(queued_ts, scheduled_ts, first_token_ts, last_token_ts):
    return types.SimpleNamespace(
        queued_ts=queued_ts,
        scheduled_ts=scheduled_ts,
        first_token_ts=first_token_ts,
        last_token_ts=last_token_ts,
    )


def test_env_unset_returns_null_writer():
    import os

    os.environ.pop("REQUEST_LOG_DIR", None)
    writer = rl.get_request_log_writer("req-a", "Tell me a story")
    assert isinstance(writer, rl.NullRequestLogWriter)


def test_full_success_flow_and_footer_alignment():
    import os

    with tempfile.TemporaryDirectory() as d:
        os.environ["REQUEST_LOG_DIR"] = d
        try:
            writer = rl.get_request_log_writer("req-a", "Tell me a story")
            assert isinstance(writer, rl.RequestLogWriter)

            writer.write_header(
                request_id="req-a",
                client_addr="127.0.0.1",
                method="POST",
                path="/v1/chat/completions",
                model="qwen3.8-27b",
                resolved_model="qwen3.8-27b",
                headers={"authorization": "Bearer abc", "content-type": "application/json"},
                sampling_params={"temperature": 0.7, "max_tokens": 100},
                model_settings={},
                raw_body='{"model": "qwen3.8-27b", "messages": []}',
                prompt_text="SYSTEM:\nbe nice\n",
                thinking_forced_open=True,
                concurrent_at_arrival=2,
            )
            # Callers pass the model's raw generated delta text only — the
            # model never generates the opening "<think>" itself (it's a
            # prompt-template artifact), so the writer synthesizes it once,
            # automatically, on the first chunk. See "Reasoning-model
            # <think> prefix" in REQUEST_LOGGING_SPEC.md.
            writer.write_chunk("reasoning...")
            writer.note_progress(
                prompt_tokens=50,
                completion_tokens=1,
                cached_tokens=10,
                metrics=_fake_metrics(100.0, 100.5, 101.0, 101.0),
                concurrency_snapshot=1,
            )
            writer.write_chunk("</think>answer")
            writer.note_progress(
                prompt_tokens=50,
                completion_tokens=20,
                cached_tokens=10,
                metrics=_fake_metrics(100.0, 100.5, 101.0, 103.0),
            )
            writer.write_footer()

            path = writer._path
            content = path.read_text()
        finally:
            del os.environ["REQUEST_LOG_DIR"]

    assert "=== VLLM REQUEST LOG ===" in content
    assert "request_id: req-a" in content
    assert "=== HEADERS ===\nauthorization: Bearer abc" in content
    assert "=== SAMPLING PARAMS ===" in content
    assert '"temperature": 0.7' in content
    assert "=== PROMPT ===\nSYSTEM:\nbe nice" in content
    assert "=== RESPONSE (streaming) ===" in content
    # note: thinking_forced_open only auto-prepends <think> on the FIRST
    # write_chunk call if the caller hasn't already; here the caller wrote
    # its own opener explicitly as the first chunk, so no duplicate.
    assert content.count("<think>") == 1
    assert "=== PERFORMANCE ===" in content
    assert "Prompt tokens" in content and ": 50" in content
    assert "Cached tokens" in content and "10 (20.0%)" in content
    assert "Concurrent @ prefill start" in content and "2 other request(s)" in content
    assert "Concurrent @ decode start" in content and "1 other request(s)" in content

    # Alignment: label column must be justified to the longest included label.
    lines = [l for l in content.splitlines() if " : " in l and "===" not in l]
    perf_lines = content.split("=== PERFORMANCE ===\n", 1)[1].splitlines()
    perf_lines = [l for l in perf_lines if l.strip()]
    colon_positions = {l.index(" : ") for l in perf_lines}
    assert len(colon_positions) == 1, f"footer rows not aligned: {perf_lines}"


def test_error_path_gives_partial_footer_not_zeroed():
    import os

    with tempfile.TemporaryDirectory() as d:
        os.environ["REQUEST_LOG_DIR"] = d
        try:
            writer = rl.get_request_log_writer("req-b", "hi")
            writer.write_header(
                request_id="req-b",
                client_addr=None,
                method="POST",
                path="/v1/completions",
                model="qwen3.8-27b",
                resolved_model="qwen3.8-27b",
                headers={},
                sampling_params={},
                model_settings={},
                raw_body="not json",
                prompt_text="hi",
                thinking_forced_open=False,
            )
            writer.write_chunk("partial output before it died")
            # Simulate the gotcha #2 case: a final zero-valued marker must
            # not be allowed to clobber the last real token counts. Callers
            # achieve this by only passing non-None values on note_progress
            # (None means "unknown for this chunk", not "zero").
            writer.note_progress(
                prompt_tokens=5,
                completion_tokens=7,
                cached_tokens=0,
                metrics=_fake_metrics(0.0, 0.0, 0.0, 0.0),
            )
            writer.write_error(RuntimeError("engine crashed"), http_status=500)
            content = writer._path.read_text()
        finally:
            del os.environ["REQUEST_LOG_DIR"]

    assert "=== ERROR ===" in content
    assert "exception_type: RuntimeError" in content
    assert "http_status: 500" in content
    assert "=== PERFORMANCE (partial) ===" in content
    assert "Completion tokens" in content and "7" in content
    assert "Completion tokens : 0" not in content


def test_close_without_footer_writes_cancelled_block_with_partial_data():
    import os

    with tempfile.TemporaryDirectory() as d:
        os.environ["REQUEST_LOG_DIR"] = d
        try:
            writer = rl.get_request_log_writer("req-c", "hi")
            writer.write_header(
                request_id="req-c",
                client_addr=None,
                method="POST",
                path="/v1/chat/completions",
                model="m",
                resolved_model="m",
                headers={},
                sampling_params={},
                model_settings={},
                raw_body="{}",
                prompt_text="hi",
                thinking_forced_open=False,
            )
            writer.write_chunk("some partial text")
            writer.note_progress(
                prompt_tokens=5, completion_tokens=3, cached_tokens=0, metrics=None
            )
            writer.close()  # no write_footer()/write_error() called first
            content = writer._path.read_text()

            # idempotent: a second close() must not raise or duplicate the block
            writer.close()
        finally:
            del os.environ["REQUEST_LOG_DIR"]

    assert "=== CANCELLED ===" in content
    assert content.count("=== CANCELLED ===") == 1
    assert "=== PERFORMANCE (partial) ===" in content
    assert "Completion tokens : 3" in content


def test_omitted_fields_not_faked_as_zero():
    writer = rl.RequestLogWriter(Path(tempfile.mktemp()))
    rows = dict(writer._build_footer_rows())
    assert "Prompt tokens" not in rows
    assert "Time to first token" not in rows
    assert "Queue wait" not in rows


def main():
    tests = [
        test_sanitize,
        test_first_n_words_stub,
        test_build_request_log_path_basic_and_collision,
        test_build_request_log_path_sanitizes_untrusted_id,
        test_render_chat_transcript,
        test_null_writer_is_all_noop,
        test_env_unset_returns_null_writer,
        test_full_success_flow_and_footer_alignment,
        test_error_path_gives_partial_footer_not_zeroed,
        test_close_without_footer_writes_cancelled_block_with_partial_data,
        test_omitted_fields_not_faked_as_zero,
    ]
    results = [check(t.__name__, t) for t in tests]
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} passed")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
