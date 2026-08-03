import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.core.config import settings
from ai_translator_api.core.readiness import model_artifact_status


class DeploymentReadinessTests(unittest.TestCase):
    def test_missing_artifacts_are_reported_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            settings, "INDICTRANS2_EN_INDIC", str(Path(directory) / "en-indic")
        ), patch.object(
            settings, "INDICTRANS2_INDIC_EN", str(Path(directory) / "indic-en")
        ), patch.object(
            settings, "WHISPER_MODEL_PATH", str(Path(directory) / "whisper")
        ), patch.object(
            settings, "OCR_MODEL_ROOT", str(Path(directory) / "paddle")
        ), patch.object(
            settings, "PIPER_VOICES_DIR", str(Path(directory) / "piper")
        ), patch.object(
            settings, "PIPER_VOICE_MAP", ""
        ):
            statuses = model_artifact_status()

        self.assertEqual(set(statuses), {"translation", "ocr", "stt", "tts"})
        self.assertTrue(all(not status.available for status in statuses.values()))
        self.assertIn("model.bin", statuses["translation"].info)
        self.assertIn("complete local model", statuses["stt"].info)

    def test_explicit_piper_map_requires_model_and_json_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "en_voice.onnx"
            model.write_bytes(b"model")
            with patch.object(settings, "PIPER_VOICES_DIR", ""), patch.object(
                settings,
                "PIPER_VOICE_MAP",
                json.dumps({"eng_Latn": str(model)}),
            ):
                status = model_artifact_status()["tts"]
            self.assertFalse(status.available)
            self.assertIn("eng_Latn", status.info)


if __name__ == "__main__":
    unittest.main()
