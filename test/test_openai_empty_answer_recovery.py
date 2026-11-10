"""Regression: an empty final answer must NOT escalate into a fatal error.

When the model emits no content, no reasoning, and no tool_calls (the classic
"reasoning-only model under think=False" pathology — DeepSeek V4.1/R1, QwQ,
o-series), ``OpenAIBinding._run_tool_loop`` used to return
``{"status": "error", "message": "[OpenAIBinding] Tool loop completed with an
EMPTY final answer ..."}``. The surrounding runtime interpreted ``status:error``
as a server fault and surfaced an HTTP 500 to the user — which killed the
whole turn.

The corrected contract:

1. On an empty first answer, ``_run_tool_loop`` retries ONCE with think=True.
2. If the retry produces content, that content is returned as a plain string.
3. If the retry is also empty, a graceful PLACEHOLDER STRING is returned
   (never a ``status:error`` dict), so a fatal 500 is impossible.

These tests stub ``_generate_one_round`` (the network layer) so they run
offline and pin only the recovery logic.
"""

from __future__ import annotations

from unittest.mock import patch

from lollms_client.llm_bindings.openai import OpenAIBinding


def _make_binding() -> OpenAIBinding:
    # Construct without hitting the network: host_address, model_name, and a
    # dummy service_key are enough. The binding lazily builds its client, and
    # the empty-answer path never calls the network client directly.
    return OpenAIBinding(model_name="test-model", service_key="sk-test")


def _call_run_tool_loop(binding: OpenAIBinding):
    """Invoke the private loop with minimal, valid arguments."""
    return binding._run_tool_loop(
        [{"role": "user", "content": "say hi"}],
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
        tools=None,
        tool_choice=None,
        extra_kwargs={"_empty_answer_recovery_test": True},
    )


def test_empty_answer_recovers_via_think_retry():
    """Empty first round → retry with think=True → returns the retry content."""
    binding = _make_binding()

    # 1st call: empty (0-content). 2nd call (the retry): real content.
    call_seq = iter([
        ("", None),                          # first round: empty
        ("Recovered content from think=True", None),  # retry: real content
    ])

    def _fake_generate_one_round(conversation, **kwargs):
        return next(call_seq)

    with patch.object(binding, "_generate_one_round", side_effect=_fake_generate_one_round):
        result = _call_run_tool_loop(binding)

    assert isinstance(result, str)
    assert "Recovered content from think=True" in result
    assert not (isinstance(result, dict) and result.get("status") == "error")


def test_empty_answer_after_retry_returns_graceful_placeholder():
    """Empty first round AND empty retry → graceful placeholder string, no 500."""
    binding = _make_binding()

    def _fake_generate_one_round(conversation, **kwargs):
        return ("", None)  # both attempts empty

    with patch.object(binding, "_generate_one_round", side_effect=_fake_generate_one_round):
        result = _call_run_tool_loop(binding)

    # A graceful placeholder STRING, never an error dict.
    assert isinstance(result, str)
    assert not isinstance(result, dict)
    assert "empty response" in result.lower()
    # Must not carry the old fatal wrappers.
    assert not result.startswith("[Generation error:")
    assert "[OpenAIBinding] Tool loop completed with an EMPTY final" not in result


def test_retry_is_called_with_think_true():
    """The recovery retry MUST set think=True — otherwise reasoning stays suppressed."""
    binding = _make_binding()

    seen_kwargs = []

    def _fake_generate_one_round(conversation, **kwargs):
        seen_kwargs.append(dict(kwargs))
        if len(seen_kwargs) == 1:
            return ("", None)               # first: empty
        return ("ok", None)                 # retry: content

    with patch.object(binding, "_generate_one_round", side_effect=_fake_generate_one_round):
        _call_run_tool_loop(binding)

    assert len(seen_kwargs) == 2, "expected exactly one retry"
    assert seen_kwargs[0].get("think") is False
    assert seen_kwargs[1].get("think") is True, (
        "recovery retry must force think=True; got "
        f"{seen_kwargs[1].get('think')!r}"
    )