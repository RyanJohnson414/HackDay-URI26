"""Small typed client for Gemma-based connection assessments."""

import json
from pydantic import BaseModel, Field, ValidationError

from app.core.config import get_settings


class GeminiClientError(RuntimeError):
    """Missing configuration, provider error, or malformed model response."""


class ConnectionAssessment(BaseModel):
    relevant: bool
    strength: float = Field(ge=0, le=1)
    title: str = Field(min_length=3, max_length=120)
    explanation: str = Field(min_length=20, max_length=1600)
    shared_concept: str = Field(min_length=2, max_length=200)


class TopicSummary(BaseModel):
    summary: str = Field(min_length=40, max_length=1800)
    key_points: list[str] = Field(max_length=5)


class GeminiClient:
    def __init__(self, *, client=None, model: str | None = None):
        self.model = model or get_settings().gemini_model
        self._client = client

    def _get_client(self):
        if self._client is None:
            key = get_settings().gemini_api_key
            if not key:
                raise GeminiClientError("GEMINI_API_KEY is required for Gemma AI features")
            try:
                from google import genai
                self._client = genai.Client(api_key=key)
            except Exception as exc:
                raise GeminiClientError(f"Could not initialize Gemini client: {exc}") from exc
        return self._client

    def assess_connection(self, text_a: str, text_b: str) -> ConnectionAssessment:
        prompt = f"""Assess whether these two cited passages express a specific, useful connection.
Only use information explicitly present in the passages. Do not invent facts, evidence, or quotations.
If no meaningful connection exists, set relevant=false, strength=0 and explain briefly.
Return only a JSON object with keys relevant (boolean), strength (number 0..1), title,
explanation, shared_concept. Explain how each passage supports the connection.

PASSAGE A:\n{text_a}\n\nPASSAGE B:\n{text_b}"""
        try:
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            raw = response.text
            if not raw:
                raise ValueError("empty model response")
            parsed = ConnectionAssessment.model_validate(json.loads(raw))
            if not parsed.relevant:
                return parsed.model_copy(update={"strength": 0.0})
            return parsed
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            raise GeminiClientError(f"Gemini returned an invalid connection assessment: {exc}") from exc
        except Exception as exc:
            raise GeminiClientError(f"Gemini request failed: {exc}") from exc

    def summarize_topic(self, topic_label: str, keywords: list[str], passages: list[str]) -> TopicSummary:
        """Summarize a topic using only its persisted source passages."""
        if not passages:
            raise GeminiClientError("This topic has no source passages to summarize")
        evidence = "\n\n".join(f"SOURCE {index + 1}:\n{text}" for index, text in enumerate(passages))
        prompt = f"""Summarize the subject represented by these stored document passages.
Use only claims supported by the supplied passages. Do not add outside facts or fabricate citations.
Return strict JSON with keys: summary (a concise, readable paragraph) and key_points (up to five short strings).
Subject label: {topic_label}
Topic keywords: {", ".join(keywords)}

SOURCE PASSAGES:
{evidence}"""
        try:
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            raw = response.text
            if not raw:
                raise ValueError("empty model response")
            return TopicSummary.model_validate(json.loads(raw))
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            raise GeminiClientError(f"Gemma returned an invalid topic summary: {exc}") from exc
        except Exception as exc:
            raise GeminiClientError(f"Gemma topic summary request failed: {exc}") from exc

    def assess_url_connection(self, url_a: str, url_b: str, *, max_chars_per_page: int = 12000
                              ) -> ConnectionAssessment:
        """Fetch two public pages and assess their extracted text with source URLs attached."""
        from app.services.webpage_reader import WebPageReadError, fetch_webpage

        if max_chars_per_page <= 0:
            raise ValueError("max_chars_per_page must be positive")
        try:
            page_a = fetch_webpage(url_a)
            page_b = fetch_webpage(url_b)
        except WebPageReadError as exc:
            raise GeminiClientError(f"Could not read source URL: {exc}") from exc
        source_a = (f"Source URL: {page_a.url}\nPage title: {page_a.title}\n\n" + page_a.text[:max_chars_per_page])
        source_b = (f"Source URL: {page_b.url}\nPage title: {page_b.title}\n\n" + page_b.text[:max_chars_per_page])
        return self.assess_connection(source_a, source_b)
