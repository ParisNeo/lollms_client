"""Regression: ``LollmsBinding`` must NOT escalate an empty final answer into
a fatal HTTP 500.

The server (a lollms server) returns HTTP 500 with the body::

    {'detail': '[OpenAIBinding] Tool loop completed with an EMPTY final answer
     (no content, no reasoning, and no tool calls on the last round).
     Likely cause: this endpoint does not support tool-aware continuation for
     the selected model, or the model emitted reasoning-only output that was
     suppressed because think=False. Enable think=True on the binding or
     switch to a tool-capable model.'}

…whenever the model emits only reasoning under ``think=False`` and the
backend agentic loop suppresses it. ``LollmsBinding`` translates this as a
fatal error today (``ERROR:ASCIIColors:[LollmsBinding] Server returned 500``).

The corrected contract:

1. ``_run_tool_loop`` retries ONCE with think=True when the final answer is
   empty; if the retry produces content, it is returned as a plain string.
2. If the retry is also empty, a graceful PLACEHOLDER STRING is returned
   (never a ``status:error`` dict), so the outer runtime cannot surface a 500.
3. ``_generate_one_round`` performs the same recovery for a plain chat turn
   (no tools requested).

These tests stub the network layer so they run offline and pin only the
recovery logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lollms_client.llm_bindings.lollms import LollmsBinding

def _make_binding() -> LollmsBinding:
    binding = LollmsBinding.__new__(LollmsBinding)
    # Minimal state required by ``_run_tool_loop`` and ``_generate_one_round``.
    binding._lollms_client_state = {}
    binding.model_name = "test-model"
    binding.service_key = "sk-test"
    binding.host_address = "http://localhost:9642"
    binding.generation_mode = "chat"
    # Wire objects referenced by _apply_thinking_params / _ensure_client.
    binding.is_vllm = False
    binding.send_thinking_parameter = False
    binding.thinking_effort_keyword = "enable_thinking"
    binding.glm_image_embedding = False
    binding.default_ctx_size = 8192
    binding.vision_enabled = False
    binding.video_enabled = False
    binding.generation_flag = True
    binding._cancel_requested = False
    binding._verify_obj = True
    binding._remote_tokenizer_healthy = False
    binding._ctx_size_failures = 0

    # Stub the SDK client so we never hit the network.
    binding.client = MagicMock()

    # _apply_thinking_params must be a real method, but to avoid heavy state
    # we stub it to a no-op that just returns the params.
    binding._apply_thinking_params = lambda params, effort, deact: params
    binding.is_cancelled = lambda: False
    binding.finalize_non_stream_output = (
        lambda content, reasoning, deactivated: content
    )
    return binding

def _run_tool_loop_sync(binding: LollmsBinding, tools):
    return binding._run_tool_loop(
        [{"role": "user", "content": "hi"}],
        n_predict=16,
        stream=False,
        temperature=0.1,
        top_k=None,
        top_p=None,
        repeat_penalty=None,
        seed=None,
        streaming_callback=None,
        think=False,
        reasoning_effort=None,
        reasoning_summary=None,
        tools=tools,
        tool_choice=None,
        extra_kwargs={},
    )

# --------------------------------------------------------------------------- #
# _run_tool_loop recovery
# --------------------------------------------------------------------------- #

def test_run_tool_loop_empty_answer_recovers_via_think_retry():
    binding = _make_binding()
    call_seq = iter([
        ("", None),                               # first round: empty
        ("Recovered content from think=True", None),  # retry: real content
    ])

    with patch.object(
        binding, "_generate_one_round",
        side_effect=lambda *a, **k: next(call_seq),
    ):
        result = _run_tool_loop_sync(binding, tools=[{"type": "function"}])

    assert isinstance(result, str)
    assert "Recovered content from think=True" in result

def test_run_tool_loop_empty_returns_graceful_placeholder():
    binding = _make_binding()

    with patch.object(
        binding, "_generate_one_round",
        side_effect=lambda *a, **k: ("", None),
    ):
        result = _run_tool_loop_sync(binding, tools=[{"type": "function"}])

    assert isinstance(result, str)
    assert not isinstance(result, dict)
    assert "empty response" in result.lower()
    assert "Tool loop completed with an EMPTY final" not in result

def test_run_tool_loop_retry_uses_think_true():
    binding = _make_binding()
    seen = []

    def _generate(conversation, **kwargs):
        seen.append(dict(kwargs))
        return ("", None) if len(seen) == 1 else ("ok", None)

    with patch.object(binding, "_generate_one_round", side_effect=_generate):
        _run_tool_loop_sync(binding, tools=[{"type": "function"}])

    assert len(seen) == 2, "expected exactly one retry"
    assert seen[0].get("think") is False
    assert seen[1].get("think") is True

# --------------------------------------------------------------------------- #
# generate_text recovery — this is the ACTUAL failing path (no client tools;
# the 500 comes from the SERVER-side tool loop and is caught here at line 1222)
# --------------------------------------------------------------------------- #

def test_generate_text_retries_with_think_on_empty_answer_500():
    """A 500 whose body says 'empty final answer' must trigger a think=True
    retry — not an immediate error dict."""
    import openai

    binding = _make_binding()
    # Two calls to chat.completions.create:
    #   1. first raises InternalServerError carrying the 'empty final answer' body
    #   2. retry (with think=True) returns a normal completion carrying content
    created_calls = []

    def _make_msg(content):
        m = MagicMock()
        m.content = content
        return m

    def _fake_create(**kwargs):
        created_calls.append(dict(kwargs))
        if len(created_calls) == 1:
            body = {
                "detail": (
                    "[OpenAIBinding] Tool loop completed with an EMPTY final "
                    "answer (no content, no reasoning, and no tool calls on "
                    "the last round). Enable think=True on the binding or "
                    "switch to a tool-capable model."
                )
            }
            resp = MagicMock()
            resp.status_code = 500
            resp.headers = {}
            resp.request = MagicMock()
            resp.json = lambda: body
            raise openai.InternalServerError(
                "Error code: 500", response=resp, body=body
            )
        completion = MagicMock()
        choice = MagicMock()
        choice.message = _make_msg("Recovered answer from think=True")
        completion.choices = [choice]
        return completion

    binding.client.chat.completions.create.side_effect = _fake_create

    result = binding.generate_text(
        prompt="hello",
        system_prompt="You are helpful.",
        think=False,
        n_predict=16,
    )

    assert isinstance(result, str)
    assert "Recovered answer from think=True" in result
    # Verify the recovery retry forced thinking ON via some signal
    assert len(created_calls) >= 2, "expected a retry after the 500"
    # The retry must NOT carry the deactivated-thinking signature (no
    # export of the empty final answer scenario under think=False).
    retry_params = created_calls[1]
    eb = retry_params.get("extra_body") or {}
    assert (
        retry_params.get("reasoning_effort") is not None
        or eb.get("thinking") is not None
        or eb.get("chat_template_kwargs", {}).get("enable_thinking") is True
    ), f"retry must force thinking ON; got {retry_params!r}"