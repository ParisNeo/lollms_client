import pytest
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.llm_bindings.openai import extract_reasoning, _StreamThinkingHandler
from lollms_client.lollms_text_processing import LollmsTextProcessor


class MockPydanticChoiceDelta:
    def __init__(self, content=None, model_extra=None, direct_reasoning=None):
        self.content = content
        self.model_extra = model_extra or {}
        if direct_reasoning:
            self.reasoning_content = direct_reasoning


class DummyLLM:
    def __init__(self):
        self.default_ctx_size = 4096
    def generate_text(self, *args, **kwargs):
        return ""
    def get_context_size(self):
        return 4096
    def count_tokens(self, text):
        return len(text) // 4


def test_extract_reasoning_direct_attr():
    delta = MockPydanticChoiceDelta(direct_reasoning="Direct reasoning string")
    assert extract_reasoning(delta) == "Direct reasoning string"


def test_extract_reasoning_model_extra():
    delta = MockPydanticChoiceDelta(model_extra={"reasoning_content": "Pydantic extra reasoning"})
    assert extract_reasoning(delta) == "Pydantic extra reasoning"


def test_extract_reasoning_dict():
    raw_dict = {"reasoning_content": "Dict-based reasoning"}
    assert extract_reasoning(raw_dict) == "Dict-based reasoning"


def test_stream_thinking_handler_dedicated_reasoning():
    chunks_received = []

    def callback(chunk: str, msg_type: MSG_TYPE):
        chunks_received.append((chunk, msg_type))
        return True

    handler = _StreamThinkingHandler(callback)
    assert handler.process_reasoning("Let me deduce this step by step.")
    assert handler.process_reasoning(" Further logical derivation.")
    assert handler.process_content("The final answer is 42.")
    final_output = handler.flush()

    assert "<think>" in final_output
    assert "</think>" in final_output
    assert "Let me deduce this step by step." in final_output
    assert "The final answer is 42." in final_output

    thought_chunks = [c for c, m in chunks_received if m == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK]
    content_chunks = [c for c, m in chunks_received if m == MSG_TYPE.MSG_TYPE_CHUNK]

    assert len(thought_chunks) >= 3
    assert "<think>\n" in thought_chunks[0]
    assert any("Let me deduce" in c for c in thought_chunks)
    assert any("The final answer is 42." in c for c in content_chunks)

    processor = LollmsTextProcessor(DummyLLM())
    clean_prompt = processor.remove_thinking_blocks(final_output)
    assert "The final answer is 42." in clean_prompt
    assert "Let me deduce this step by step" not in clean_prompt
    assert "<think>" not in clean_prompt
    assert "</think>" not in clean_prompt


def test_stream_thinking_handler_in_content_tags():
    chunks_received = []

    def callback(chunk: str, msg_type: MSG_TYPE):
        chunks_received.append((chunk, msg_type))
        return True

    handler = _StreamThinkingHandler(callback)
    assert handler.process_content("<think>\nInternal calculation")
    assert handler.process_content(" in progress.\n</think>\nActual response.")
    final_output = handler.flush()

    assert final_output.startswith("<think>")
    assert "</think>" in final_output
    assert "Actual response." in final_output

    thought_chunks = [c for c, m in chunks_received if m == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK]
    content_chunks = [c for c, m in chunks_received if m == MSG_TYPE.MSG_TYPE_CHUNK]

    assert any("Internal calculation" in c for c in thought_chunks)
    assert any("Actual response." in c for c in content_chunks)

    processor = LollmsTextProcessor(DummyLLM())
    clean_prompt = processor.remove_thinking_blocks(final_output)
    assert clean_prompt.strip() == "Actual response."