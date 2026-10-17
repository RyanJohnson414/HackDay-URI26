import json
from types import SimpleNamespace

from app.services import gemini_client as module
from app.services.gemini_client import GeminiClient


class RecordingModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        payload = {
            "title": "Local signal systems",
            "domain": "Systems science",
            "tags": ["local signals", "adaptation"],
            "sections": [{
                "title": "Local coordination",
                "summary": "The source describes coordination through nearby signals.",
                "page_start": 1,
                "page_end": 1,
            }],
        }
        return SimpleNamespace(text=json.dumps(payload))


def test_document_analysis_uses_configured_model_for_every_parsed_text_batch(monkeypatch):
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(gemini_model="env-gemini-model"))
    models = RecordingModels()
    client = GeminiClient(client=SimpleNamespace(models=models))
    text = "first source passage. " + ("x" * 30_001) + " final source passage."

    result = client.analyze_document("source.txt", [(1, text)])

    assert len(models.calls) == 2
    assert all(call["model"] == "env-gemini-model" for call in models.calls)
    assert all(text[offset:offset + 30_000] in call["contents"]
               for offset, call in zip((0, 30_000), models.calls))
    assert result.title == "Local signal systems"
    assert result.tags == ["local signals", "adaptation"]
    assert result.sections[0].title == "Local coordination"
