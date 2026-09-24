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


def test_glm_image_embedding_normalization():
    from lollms_client.llm_bindings.openai import normalize_image_input

    url_block = normalize_image_input("https://example.com/test.png", glm_format=True)
    assert url_block["type"] == "image_url"
    assert url_block["image_url"]["url"] == "https://example.com/test.png"

    dict_url_block = normalize_image_input({"url": "http://img.site/pic.jpg"}, glm_format=True)
    assert dict_url_block["type"] == "image_url"
    assert dict_url_block["image_url"]["url"] == "http://img.site/pic.jpg"

    b64_block = normalize_image_input("aW1nZGF0YQ==", glm_format=True)
    assert b64_block["type"] == "image_url"
    assert b64_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_glm_image_embedding_profile_propagation():
    from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile

    client = LollmsClient(
        llm_binding_profiles={
            "mock_openai": LollmsBindingProfile(
                name="mock_openai",
                binding_name="openai",
                binding_config={"host_address": "http://localhost:8000/v1"}
            )
        },
        llm_model_profiles={
            "glm_flash": LollmsModelProfile(
                name="glm_flash",
                binding_profile_name="mock_openai",
                model_name="zai-org/GLM-5.3-Flash",
                vision_enabled=True,
                glm_image_embedding=True,
                is_default=True
            )
        }
    )

    assert client.has_vision_capability() is True
    assert getattr(client.llm, "glm_image_embedding", False) is True


def test_translate_reasoning_effort_mapping():
    from lollms_client.lollms_llm_binding import LollmsLLMBinding

    openai_levels = ["low", "medium", "high"]
    glm_levels = ["low", "high", "max"]
    binary_levels = ["off", "on"]

    assert LollmsLLMBinding.translate_reasoning_effort("max", openai_levels) == "high"
    assert LollmsLLMBinding.translate_reasoning_effort("medium", openai_levels) == "medium"
    assert LollmsLLMBinding.translate_reasoning_effort("minimal", openai_levels) == "low"

    assert LollmsLLMBinding.translate_reasoning_effort("max", glm_levels) == "max"
    assert LollmsLLMBinding.translate_reasoning_effort("medium", glm_levels) == "high"
    assert LollmsLLMBinding.translate_reasoning_effort("low", glm_levels) == "low"

    assert LollmsLLMBinding.translate_reasoning_effort("high", binary_levels) == "on"
    assert LollmsLLMBinding.translate_reasoning_effort("none", binary_levels) == "off"
    assert LollmsLLMBinding.translate_reasoning_effort("none", openai_levels) is None

    assert LollmsLLMBinding.translate_reasoning_effort(True, glm_levels) == "high"
    assert LollmsLLMBinding.translate_reasoning_effort(False, glm_levels) is None


def test_model_profile_supported_efforts_propagation():
    from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile

    client = LollmsClient(
        llm_binding_profiles={
            "mock_conn": LollmsBindingProfile(
                name="mock_conn",
                binding_name="openai",
                binding_config={"host_address": "http://localhost:8000/v1"}
            )
        },
        llm_model_profiles={
            "custom_o3": LollmsModelProfile(
                name="custom_o3",
                binding_profile_name="mock_conn",
                model_name="o3-mini",
                supported_reasoning_efforts=["low", "medium", "high"],
                is_default=True
            )
        }
    )

    assert client.llm.supported_reasoning_efforts == ["low", "medium", "high"]
    assert client.llm.get_effective_reasoning_effort(reasoning_effort="max") == "high"


def test_normalize_video_input():
    from lollms_client.llm_bindings.openai import normalize_video_input

    url_block = normalize_video_input("https://example.com/demo.mp4")
    assert url_block["type"] == "video_url"
    assert url_block["video_url"]["url"] == "https://example.com/demo.mp4"

    dict_block = normalize_video_input({"url": "http://vid.site/sample.webm"})
    assert dict_block["type"] == "video_url"
    assert dict_block["video_url"]["url"] == "http://vid.site/sample.webm"

    b64_block = normalize_video_input("dmlkZW9kYXRh")
    assert b64_block["type"] == "video_url"
    assert b64_block["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_video_capability_profile_propagation():
    from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile

    client = LollmsClient(
        llm_binding_profiles={
            "mock_conn": LollmsBindingProfile(
                name="mock_conn",
                binding_name="openai",
                binding_config={"host_address": "http://localhost:8000/v1"}
            )
        },
        llm_model_profiles={
            "qwen_video": LollmsModelProfile(
                name="qwen_video",
                binding_profile_name="mock_conn",
                model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                video_enabled=True,
                is_default=True
            )
        }
    )

    assert client.has_video_capability() is True
    assert client.has_vision_capability() is True
    assert getattr(client.llm, "video_enabled", False) is True


def test_translate_reasoning_effort_mapping():
    from lollms_client.lollms_llm_binding import LollmsLLMBinding

    openai_levels = ["low", "medium", "high"]
    glm_levels = ["low", "high", "max"]
    binary_levels = ["off", "on"]

    assert LollmsLLMBinding.translate_reasoning_effort("max", openai_levels) == "high"
    assert LollmsLLMBinding.translate_reasoning_effort("medium", openai_levels) == "medium"
    assert LollmsLLMBinding.translate_reasoning_effort("minimal", openai_levels) == "low"

    assert LollmsLLMBinding.translate_reasoning_effort("max", glm_levels) == "max"
    assert LollmsLLMBinding.translate_reasoning_effort("medium", glm_levels) == "high"
    assert LollmsLLMBinding.translate_reasoning_effort("low", glm_levels) == "low"

    assert LollmsLLMBinding.translate_reasoning_effort("high", binary_levels) == "on"
    assert LollmsLLMBinding.translate_reasoning_effort("none", binary_levels) == "off"
    assert LollmsLLMBinding.translate_reasoning_effort("none", openai_levels) is None

    assert LollmsLLMBinding.translate_reasoning_effort(True, glm_levels) == "high"
    assert LollmsLLMBinding.translate_reasoning_effort(False, glm_levels) is None


def test_model_profile_supported_efforts_propagation():
    from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile

    client = LollmsClient(
        llm_binding_profiles={
            "mock_conn": LollmsBindingProfile(
                name="mock_conn",
                binding_name="openai",
                binding_config={"host_address": "http://localhost:8000/v1"}
            )
        },
        llm_model_profiles={
            "custom_o3": LollmsModelProfile(
                name="custom_o3",
                binding_profile_name="mock_conn",
                model_name="o3-mini",
                supported_reasoning_efforts=["low", "medium", "high"],
                is_default=True
            )
        }
    )

    assert client.llm.supported_reasoning_efforts == ["low", "medium", "high"]
    assert client.llm.get_effective_reasoning_effort(reasoning_effort="max") == "high"