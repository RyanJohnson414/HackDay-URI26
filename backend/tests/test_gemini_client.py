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


def test_connection_assessment_uses_configured_model_and_fifty_percent_threshold(monkeypatch):
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(gemini_model="env-gemma-model"))

    class ConnectionModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(text=json.dumps({
                "relevant": True,
                "strength": 0.8,
                "title": "Shared local feedback",
                "explanation": "Both passages describe local feedback shaping behavior.",
                "shared_concept": "local feedback",
                "evidence_a": "Local signals guide colony behavior.",
                "evidence_b": "Local feedback guides routing behavior.",
            }))

    models = ConnectionModels()
    result = GeminiClient(client=SimpleNamespace(models=models)).assess_connection(
        "SOURCE PASSAGE:\nLocal signals guide colony behavior.",
        "SOURCE PASSAGE:\nLocal feedback guides routing behavior.",
    )

    prompt = models.calls[0]["contents"]
    assert "calibrated 0..1 scale" in prompt
    assert "strength >= 0.50" in prompt
    assert "evidence_a" in prompt and "evidence_b" in prompt
    assert models.calls[0]["model"] == "env-gemma-model"
    assert result.strength == 0.8
    assert result.evidence_a == "Local signals guide colony behavior."


def test_connection_assessment_accepts_list_wrapped_response_and_empty_shared_concept():
    class ConnectionModels:
        def generate_content(self, **_kwargs):
            return SimpleNamespace(text=json.dumps([{
                "relevant": False,
                "strength": 0.0,
                "title": "No meaningful connection",
                "explanation": "The passages do not establish a meaningful shared mechanism.",
                "shared_concept": "",
            }]))

    result = GeminiClient(client=SimpleNamespace(models=ConnectionModels())).assess_connection(
        "A passage about one subject.", "A passage about another subject.",
    )

    assert result.relevant is False
    assert result.strength == 0.0
    assert result.shared_concept == ""


def test_document_analysis_accepts_single_analysis_wrapped_in_a_list(monkeypatch):
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(gemini_model="test-model"))
    models = RecordingModels()
    original_generate = models.generate_content

    def list_response(**kwargs):
        response = original_generate(**kwargs)
        return SimpleNamespace(text=json.dumps([json.loads(response.text)]))

    models.generate_content = list_response
    result = GeminiClient(client=SimpleNamespace(models=models)).analyze_document(
        "source.txt", [(1, "A source passage about local adaptation.")],
    )

    assert result.title == "Local signal systems"
