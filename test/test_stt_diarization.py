import base64
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lollms_client.lollms_stt_binding import LollmsSTTBinding
from lollms_client.stt_bindings.whisper import WhisperSTTBinding
from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile


class DummySTTBinding(LollmsSTTBinding):
    """Minimal test binding without native diarization to test base fallback."""
    def __init__(self):
        super().__init__(binding_name="dummy_stt")

    def transcribe_audio(self, audio_source, model=None, **kwargs):
        return "Hello world this is a test."

    def list_models(self, **kwargs):
        return ["dummy-1"]


class TestSTTDiarization(unittest.TestCase):

    def test_base_fallback_without_native_diarization(self):
        binding = DummySTTBinding()
        # Case 1: no participants provided -> defaults to Speaker 1
        turns = binding.transcribe_audio_with_diarization(b"fake_audio_bytes")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["speaker"], "Speaker 1")
        self.assertEqual(turns[0]["text"], "Hello world this is a test.")

        # Case 2: participants provided -> first participant assigned
        turns_named = binding.transcribe_audio_with_diarization(
            b"fake_audio_bytes",
            participants=["Alice", "Bob"]
        )
        self.assertEqual(len(turns_named), 1)
        self.assertEqual(turns_named[0]["speaker"], "Alice")
        self.assertEqual(turns_named[0]["text"], "Hello world this is a test.")

    @patch("requests.Session.post")
    def test_whisper_binding_transcribe_audio_with_diarization(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "turns": [
                {"speaker": "Alice", "start": 0.0, "end": 2.5, "text": "Hi everyone."},
                {"speaker": "Bob", "start": 2.6, "end": 5.0, "text": "Good morning Alice."},
                {"speaker": "Speaker 3", "start": 5.2, "end": 7.0, "text": "Hello team."}
            ],
            "text": "[Alice]: Hi everyone.\n[Bob]: Good morning Alice.\n[Speaker 3]: Hello team."
        }
        mock_post.return_value = mock_response

        binding = WhisperSTTBinding(auto_start_server=False)

        with patch.object(binding, "is_server_running", return_value=True):
            turns = binding.transcribe_audio_with_diarization(
                audio_source=b"fake_audio_bytes",
                participants=["Alice", "Bob"],
                voice_samples={"Alice": b"alice_voice_bytes"},
                model="base"
            )

            self.assertEqual(len(turns), 3)
            self.assertEqual(turns[0]["speaker"], "Alice")
            self.assertEqual(turns[1]["speaker"], "Bob")
            self.assertEqual(turns[2]["speaker"], "Speaker 3")
            self.assertEqual(turns[1]["text"], "Good morning Alice.")

            # Validate that request payload contains encoded parameters
            call_kwargs = mock_post.call_args[1]
            data = call_kwargs["json"]
            self.assertIn("audio_b64", data)
            self.assertEqual(data["participants"], ["Alice", "Bob"])
            self.assertIn("Alice", data["voice_samples"])

    def test_lollms_client_delegation(self):
        mock_stt = MagicMock()
        mock_stt.transcribe_audio_with_diarization.return_value = [
            {"speaker": "ParisNeo", "start": 0.0, "end": 3.0, "text": "Welcome to LoLLMS."}
        ]

        client = LollmsClient(
            stt_binding_profiles={
                "local_whisper": LollmsBindingProfile(
                    name="local_whisper",
                    binding_name="whisper",
                    is_default=True
                )
            },
            stt_model_profiles={
                "whisper_base": LollmsModelProfile(
                    name="whisper_base",
                    binding_profile_name="local_whisper",
                    model_name="base",
                    is_default=True
                )
            }
        )
        client.stt = mock_stt

        res = client.transcribe_audio_with_diarization(
            audio_source=b"dummy",
            participants=["ParisNeo"]
        )

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["speaker"], "ParisNeo")
        mock_stt.transcribe_audio_with_diarization.assert_called_once_with(
            audio_source=b"dummy",
            participants=["ParisNeo"]
        )


if __name__ == "__main__":
    unittest.main()