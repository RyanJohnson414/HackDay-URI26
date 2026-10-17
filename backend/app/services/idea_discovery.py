"""Two-stage cross-domain idea discovery over a small SQLite note corpus.

Stage one embeds and clusters notes, scores cross-cluster pairs, and keeps
non-obvious candidates. Stage two asks a configured Gemma model to verify
evidence. Only validated, sufficiently confident edges are persisted.

Run from ``backend`` with ``python -m app.services.idea_discovery --input notes``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy.engine import make_url


DEFAULT_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_DATABASE_URL = "sqlite:///./knowledge_graph.db"
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
RELATIONSHIP_TYPES = {"analogical", "structural", "causal", "none"}
NOTE_NAMESPACE = uuid.UUID("e2a345cc-b004-4d43-8dc0-9c28c82c324b")


class IdeaDiscoveryError(RuntimeError):
    """Base error for input, embedding, persistence, or inference failures."""


@dataclass
class Note:
    id: str
    title: str
    domain: str
    content: str
    cluster_id: int | None = None
    embedding: list[float] | None = None


@dataclass(frozen=True)
class Weights:
    w_s: float = 0.25
    w_b: float = 0.40
    w_n: float = 0.20
    w_v: float = 0.15
    w_evidence: float = 0.25


@dataclass(frozen=True)
class Candidate:
    note_a: Note
    note_b: Note
    semantic: float
    novelty: float
    bridge: float
    variance: float
    preliminary_score: float


@dataclass(frozen=True)
class Verification:
    relationship_type: str
    shared_principle: str
    evidence_a: str
    evidence_b: str
    confidence: float


@dataclass(frozen=True)
class DiscoveredConnection:
    source_id: str
    target_id: str
    composite_score: float
    relationship_type: str
    shared_principle: str
    evidence_a: str
    evidence_b: str
    confidence: float
    verified_by: str


def _environment() -> dict[str, str]:
    """Read backend/.env without overriding explicitly set process variables."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    values: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    values.update(os.environ)
    return values


def database_path(database_url: str | None = None) -> str:
    """Parse a SQLAlchemy SQLite URL into the filename accepted by sqlite3."""
    url = make_url(database_url or _environment().get("DATABASE_URL", DEFAULT_DATABASE_URL))
    if url.get_backend_name() != "sqlite":
        raise IdeaDiscoveryError("idea_discovery supports SQLite DATABASE_URL values only")
    if url.database in (None, ""):
        return ":memory:"
    path = url.database
    if path != ":memory:":
        path = str(Path(path).expanduser())
        if not Path(path).is_absolute():
            path = str((Path.cwd() / path).resolve())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class IdeaStore:
    """Minimal SQLite data layer with idempotent note and edge upserts."""

    def __init__(self, database_url: str | None = None):
        self.path = database_path(database_url)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                domain TEXT NOT NULL,
                content TEXT NOT NULL,
                cluster_id INTEGER,
                embedding BLOB
            );
            CREATE TABLE IF NOT EXISTS connections (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                composite_score REAL NOT NULL,
                relationship_type TEXT NOT NULL,
                shared_principle TEXT NOT NULL,
                evidence_a TEXT NOT NULL,
                evidence_b TEXT NOT NULL,
                confidence REAL NOT NULL,
                verified_by TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES notes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES notes(id) ON DELETE CASCADE
            );
        """)
        self.connection.commit()

    def upsert_notes(self, notes: Iterable[Note]) -> int:
        rows = []
        for note in notes:
            vector = json.dumps(note.embedding, separators=(",", ":")) if note.embedding is not None else None
            rows.append((note.id, note.title, note.domain, note.content, note.cluster_id, vector))
        with self.connection:
            self.connection.executemany("""
                INSERT INTO notes(id,title,domain,content,cluster_id,embedding)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title, domain=excluded.domain,
                    content=excluded.content, cluster_id=excluded.cluster_id,
                    embedding=COALESCE(excluded.embedding, notes.embedding)
            """, rows)
        return len(rows)

    def load_notes(self) -> list[Note]:
        result = []
        for row in self.connection.execute("SELECT * FROM notes ORDER BY id"):
            try:
                embedding = json.loads(row["embedding"]) if row["embedding"] else None
            except (TypeError, json.JSONDecodeError) as exc:
                raise IdeaDiscoveryError(f"Note {row['id']} has an invalid stored embedding") from exc
            result.append(Note(row["id"], row["title"], row["domain"], row["content"],
                               row["cluster_id"], embedding))
        return result

    def upsert_connection(self, edge: DiscoveredConnection) -> None:
        with self.connection:
            self.connection.execute("""
                INSERT INTO connections(source_id,target_id,composite_score,relationship_type,
                    shared_principle,evidence_a,evidence_b,confidence,verified_by)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_id,target_id) DO UPDATE SET
                    composite_score=excluded.composite_score,
                    relationship_type=excluded.relationship_type,
                    shared_principle=excluded.shared_principle,
                    evidence_a=excluded.evidence_a,evidence_b=excluded.evidence_b,
                    confidence=excluded.confidence,verified_by=excluded.verified_by
            """, (edge.source_id, edge.target_id, edge.composite_score, edge.relationship_type,
                  edge.shared_principle, edge.evidence_a, edge.evidence_b, edge.confidence,
                  edge.verified_by))

    def export_jsonld(self, path: str | Path) -> Path:
        nodes = [dict(row) for row in self.connection.execute(
            "SELECT id,title,domain,content,cluster_id FROM notes ORDER BY id")]
        edges = [dict(row) for row in self.connection.execute(
            "SELECT * FROM connections ORDER BY source_id,target_id")]
        graph = {
            "@context": {"id": "@id", "type": "@type", "source": {"@id": "source_id", "@type": "@id"},
                         "target": {"@id": "target_id", "@type": "@id"}},
            "@graph": ([{"@id": f"note:{n['id']}", "@type": "Note", **n} for n in nodes]
                       + [{"@id": f"connection:{e['source_id']}:{e['target_id']}",
                           "@type": "Connection", **e} for e in edges]),
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        return output

    def export_graphml(self, path: str | Path) -> Path:
        root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
        for key, scope, name, kind in (
            ("title", "node", "title", "string"), ("domain", "node", "domain", "string"),
            ("composite_score", "edge", "composite_score", "double"),
            ("relationship_type", "edge", "relationship_type", "string"),
            ("shared_principle", "edge", "shared_principle", "string"),
            ("confidence", "edge", "confidence", "double"),
        ):
            ET.SubElement(root, "key", id=key, **{"for": scope, "attr.name": name, "attr.type": kind})
        graph = ET.SubElement(root, "graph", edgedefault="undirected")
        for row in self.connection.execute("SELECT id,title,domain FROM notes ORDER BY id"):
            node = ET.SubElement(graph, "node", id=str(row["id"]))
            for key in ("title", "domain"):
                ET.SubElement(node, "data", key=key).text = str(row[key])
        for row in self.connection.execute("SELECT * FROM connections ORDER BY source_id,target_id"):
            edge = ET.SubElement(graph, "edge", source=row["source_id"], target=row["target_id"])
            for key in ("composite_score", "relationship_type", "shared_principle", "confidence"):
                ET.SubElement(edge, "data", key=key).text = str(row[key])
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
        return output

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "IdeaStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _stable_id(identity: str) -> str:
    return str(uuid.uuid5(NOTE_NAMESPACE, identity))


def _note_from_mapping(item: dict[str, Any], identity: str) -> Note:
    title = str(item.get("title") or "").strip()
    content = str(item.get("content") or "").strip()
    if not title or not content:
        raise IdeaDiscoveryError(f"Note in {identity} must have non-empty title and content")
    raw_vector = item.get("embedding")
    try:
        vector = [float(value) for value in raw_vector] if raw_vector is not None else None
        cluster = int(item["cluster_id"]) if item.get("cluster_id") is not None else None
    except (TypeError, ValueError) as exc:
        raise IdeaDiscoveryError(f"Note {title!r} has invalid cluster_id or embedding") from exc
    return Note(str(item.get("id") or _stable_id(identity)), title,
                str(item.get("domain") or "Unspecified").strip(), content, cluster, vector)


def ingest_json(path: str | Path) -> list[Note]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdeaDiscoveryError(f"Cannot read JSON notes from {source}: {exc}") from exc
    items = data.get("notes") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise IdeaDiscoveryError("JSON input must be a list of notes or an object with a 'notes' list")
    if any(not isinstance(item, dict) for item in items):
        raise IdeaDiscoveryError("Every JSON note must be an object")
    return [_note_from_mapping(item, f"{source.resolve()}:{i}")
            for i, item in enumerate(items)]


def _parse_markdown(path: Path) -> Note:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IdeaDiscoveryError(f"Cannot read Markdown note {path}: {exc}") from exc
    title, domain, cluster = path.stem.replace("_", " "), "Unspecified", None
    lines = raw.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            for line in lines[1:end]:
                match = re.match(r"^\s*(title|domain|cluster_id)\s*:\s*(.*?)\s*$", line, re.I)
                if match:
                    key, value = match.group(1).lower(), match.group(2).strip().strip("\"'")
                    if key == "title": title = value
                    elif key == "domain": domain = value
                    else:
                        try: cluster = int(value)
                        except ValueError as exc: raise IdeaDiscoveryError(f"Invalid cluster_id in {path}") from exc
            lines = lines[end + 1:]
    body = "\n".join(lines).strip()
    if body.startswith("# "):
        heading, _, remainder = body.partition("\n")
        if title == path.stem.replace("_", " "):
            title = heading[2:].strip() or title
        body = remainder.strip()
    return _note_from_mapping({"title": title, "domain": domain, "content": body,
                               "cluster_id": cluster}, str(path.resolve()))


def ingest_markdown(path: str | Path) -> list[Note]:
    source = Path(path)
    files = sorted(source.rglob("*.md")) if source.is_dir() else [source]
    if not files:
        raise IdeaDiscoveryError(f"No Markdown files found under {source}")
    markdown_files = [file for file in files if file.is_file() and file.suffix.lower() == ".md"]
    if not markdown_files:
        raise IdeaDiscoveryError(f"No Markdown files found under {source}")
    return [_parse_markdown(file) for file in markdown_files]


def ingest_input(path: str | Path) -> list[Note]:
    source = Path(path)
    if source.is_dir() or source.suffix.lower() in {".md", ".markdown"}:
        return ingest_markdown(source)
    if source.suffix.lower() == ".json":
        return ingest_json(source)
    raise IdeaDiscoveryError("Input must be a JSON file, Markdown file, or directory of Markdown files")


def ingest_into_store(store: IdeaStore, path: str | Path) -> int:
    """Read a supported input source and upsert its notes into the store."""
    return store.upsert_notes(ingest_input(path))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise IdeaDiscoveryError("Embedding vectors must have the same non-zero dimension")
    norm_l = math.sqrt(sum(float(x) ** 2 for x in left))
    norm_r = math.sqrt(sum(float(x) ** 2 for x in right))
    if norm_l == 0 or norm_r == 0:
        return 0.0
    score = sum(float(a) * float(b) for a, b in zip(left, right)) / (norm_l * norm_r)
    return max(-1.0, min(1.0, score))


class Embedder:
    """Lazy embedding adapter: Google GenAI by default or optional SentenceTransformers."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        env = _environment()
        self.provider = (provider or env.get("IDEA_EMBEDDING_PROVIDER", "google")).lower()
        self.model = model or env.get("IDEA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._model = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "sentence-transformers":
            try:
                from importlib import import_module

                SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
                if self._model is None:
                    self._model = SentenceTransformer(self.model or "all-MiniLM-L6-v2")
                return [[float(v) for v in row] for row in self._model.encode(list(texts), normalize_embeddings=False)]
            except ImportError as exc:
                raise IdeaDiscoveryError("Install sentence-transformers to use IDEA_EMBEDDING_PROVIDER=sentence-transformers") from exc
            except Exception as exc:
                raise IdeaDiscoveryError(f"SentenceTransformers embedding failed: {exc}") from exc
        if self.provider != "google":
            raise IdeaDiscoveryError("IDEA_EMBEDDING_PROVIDER must be 'google' or 'sentence-transformers'")
        env = _environment()
        key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
        if not key:
            raise IdeaDiscoveryError("GEMINI_API_KEY is required for Google embeddings")
        try:
            from google import genai
            client = genai.Client(api_key=key)
            vectors: list[list[float]] = []
            for text in texts:
                response = client.models.embed_content(model=self.model, contents=text)
                vectors.append([float(x) for x in response.embeddings[0].values])
            return vectors
        except Exception as exc:
            raise IdeaDiscoveryError(f"Google embedding request failed: {exc}") from exc


def ensure_embeddings(notes: list[Note], embedder: Embedder | None = None) -> None:
    missing = [i for i, note in enumerate(notes) if note.embedding is None]
    if not missing:
        return
    service = embedder or Embedder()
    vectors = service.embed([notes[i].title + "\nDomain: " + notes[i].domain + "\n" + notes[i].content
                             for i in missing])
    if len(vectors) != len(missing):
        raise IdeaDiscoveryError("Embedding provider returned an unexpected number of vectors")
    for index, vector in zip(missing, vectors):
        notes[index].embedding = vector
    dimensions = {len(note.embedding or []) for note in notes}
    if len(dimensions) != 1 or 0 in dimensions:
        raise IdeaDiscoveryError("All note embeddings must use the same non-zero dimension")


def assign_clusters(notes: list[Note], *, random_state: int = 42) -> dict[int, list[float]]:
    if not notes:
        return {}
    if any(note.embedding is None for note in notes):
        raise IdeaDiscoveryError("Embeddings are required before clustering")
    dimension = len(notes[0].embedding or [])
    if any(len(note.embedding or []) != dimension for note in notes):
        raise IdeaDiscoveryError("All note embeddings must have the same dimension")
    missing = [note for note in notes if note.cluster_id is None]
    assigned = [note for note in notes if note.cluster_id is not None]
    if missing and not assigned:
        if len(notes) == 1:
            notes[0].cluster_id = 0
        else:
            try:
                KMeans = __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans
                import numpy as np
            except ImportError as exc:
                raise IdeaDiscoveryError("Install scikit-learn and numpy to cluster notes") from exc
            matrix = np.asarray([note.embedding for note in notes], dtype=float)
            k = min(len(notes), max(2, round(math.sqrt(len(notes) / 2))))
            labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(matrix)
            for note, label in zip(notes, labels): note.cluster_id = int(label)
    elif missing:
        existing: dict[int, list[float]] = {}
        for note in assigned:
            existing.setdefault(int(note.cluster_id), []).append(note)  # type: ignore[arg-type]
        centroids = {key: [sum(n.embedding[j] for n in group) / len(group) for j in range(dimension)]
                     for key, group in existing.items()}
        for note in missing:
            note.cluster_id = max(centroids, key=lambda key: cosine(note.embedding or [], centroids[key]))
    groups: dict[int, list[Note]] = {}
    for note in notes:
        groups.setdefault(int(note.cluster_id), []).append(note)  # type: ignore[arg-type]
    return {key: [sum(note.embedding[j] for note in group) / len(group) for j in range(dimension)]
            for key, group in groups.items()}


def score_candidates(notes: list[Note], weights: Weights = Weights(), *,
                     min_bridge: float = 0.30, max_semantic: float = 0.70) -> list[Candidate]:
    if any(not math.isfinite(value) or value < 0 for value in
           (weights.w_s, weights.w_b, weights.w_n, weights.w_v, weights.w_evidence)):
        raise ValueError("All scoring weights must be finite and non-negative")
    centroids = assign_clusters(notes)
    if len(centroids) < 2:
        return []
    similarities: dict[tuple[str, str], float] = {}
    for i, a in enumerate(notes):
        for b in notes[i + 1:]:
            similarities[(a.id, b.id)] = cosine(a.embedding or [], b.embedding or [])

    order = {id(note): index for index, note in enumerate(notes)}

    def pair_sim(a: Note, b: Note) -> float:
        key = (a.id, b.id) if order[id(a)] < order[id(b)] else (b.id, a.id)
        return similarities[key]

    def novelty(note: Note) -> float:
        others = [other for other in notes if other.id != note.id]
        return 1.0 - max((cosine(note.embedding or [], other.embedding or []) for other in others), default=0.0)

    dispersions = {note.id: (1.0 - cosine(note.embedding or [], centroids[int(note.cluster_id)])) / 2.0
                   for note in notes}
    candidates: list[Candidate] = []
    for i, a in enumerate(notes):
        for b in notes[i + 1:]:
            if a.cluster_id == b.cluster_id:
                continue
            semantic = pair_sim(a, b)
            bridge = cosine(a.embedding or [], centroids[int(b.cluster_id)]) * cosine(
                b.embedding or [], centroids[int(a.cluster_id)])
            if bridge <= min_bridge or semantic >= max_semantic:
                continue
            combined_novelty = (novelty(a) + novelty(b)) / 2.0
            variance = (dispersions[a.id] + dispersions[b.id]) / 2.0
            score = (weights.w_s * semantic + weights.w_b * bridge
                     + weights.w_n * combined_novelty + weights.w_v * variance)
            candidates.append(Candidate(a, b, semantic, combined_novelty, bridge, variance, score))
    return sorted(candidates, key=lambda candidate: candidate.preliminary_score, reverse=True)


class GemmaVerifier:
    """Structured verifier using Google GenAI or an OpenAI-compatible local endpoint."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        env = _environment()
        self.model = model or env.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.base_url = base_url or env.get("IDEA_LLM_BASE_URL", "")
        self.api_key = env.get("GEMINI_API_KEY", "")

    def verify(self, a: Note, b: Note) -> Verification:
        prompt = f"""You verify candidate cross-domain connections using only the supplied notes.
Return exactly one JSON object with these keys: relationship_type (one of analogical,
structural, causal, none), shared_principle (string), evidence_a (string), evidence_b
(string), confidence (number 0.0 to 1.0). Evidence fields must be exact short excerpts
copied verbatim from the matching note. Use relationship_type none and confidence 0
when evidence does not support a meaningful shared principle. Do not infer or invent facts.

NOTE A | title: {a.title} | domain: {a.domain}
{a.content}

NOTE B | title: {b.title} | domain: {b.domain}
{b.content}"""
        try:
            raw = self._local_generate(prompt) if self.base_url else self._google_generate(prompt)
            payload = json.loads(raw)
            if not isinstance(payload, dict) or set(payload) != {
                    "relationship_type", "shared_principle", "evidence_a", "evidence_b", "confidence"}:
                raise ValueError("response keys do not match the required JSON schema")
            kind = payload["relationship_type"]
            confidence = float(payload["confidence"])
            principle, ev_a, ev_b = (payload[k] for k in ("shared_principle", "evidence_a", "evidence_b"))
            if kind not in RELATIONSHIP_TYPES or not all(isinstance(x, str) for x in (principle, ev_a, ev_b)):
                raise ValueError("invalid relationship type or string fields")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")
            if kind != "none":
                if not principle.strip() or not ev_a.strip() or not ev_b.strip():
                    raise ValueError("a supported connection requires a principle and evidence from both notes")
                if not _contains_verbatim(a.content, ev_a) or not _contains_verbatim(b.content, ev_b):
                    raise ValueError("evidence must be verbatim text from its source note")
            return Verification(kind, principle.strip(), ev_a.strip(), ev_b.strip(), confidence)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise IdeaDiscoveryError(f"Gemma returned an invalid verification: {exc}") from exc

    def _google_generate(self, prompt: str) -> str:
        if not self.api_key:
            raise IdeaDiscoveryError("Set GEMINI_API_KEY or configure IDEA_LLM_BASE_URL for local inference")
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model, contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            if not response.text:
                raise ValueError("empty response")
            return response.text
        except Exception as exc:
            raise IdeaDiscoveryError(f"Gemma Google GenAI request failed: {exc}") from exc

    def _local_generate(self, prompt: str) -> str:
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/v1/chat/completions" if not endpoint.endswith("/v1") else "/chat/completions"
        body = json.dumps({"model": self.model, "temperature": 0,
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": "Return only valid JSON."},
                                        {"role": "user", "content": prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        token = _environment().get("IDEA_LLM_API_KEY")
        if token: headers["Authorization"] = f"Bearer {token}"
        request = Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (HTTPError, URLError, OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise IdeaDiscoveryError(f"Local Gemma inference request failed: {exc}") from exc


def _contains_verbatim(source: str, evidence: str) -> bool:
    normalize = lambda value: " ".join(value.split()).casefold()
    return bool(evidence.strip()) and normalize(evidence) in normalize(source)


def discover_ideas(store: IdeaStore, *, top_k: int = 20, min_confidence: float = 0.60,
                   weights: Weights = Weights(), verifier: GemmaVerifier | None = None,
                   embedder: Embedder | None = None) -> list[DiscoveredConnection]:
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    notes = store.load_notes()
    ensure_embeddings(notes, embedder)
    store.upsert_notes(notes)
    candidates = score_candidates(notes, weights)[:top_k]
    model = verifier or GemmaVerifier()
    edges: list[DiscoveredConnection] = []
    for candidate in candidates:
        result = model.verify(candidate.note_a, candidate.note_b)
        if result.relationship_type == "none" or result.confidence < min_confidence:
            continue
        # Canonical ordering makes the composite primary key deterministic.
        first, second = sorted((candidate.note_a, candidate.note_b), key=lambda note: note.id)
        if first is candidate.note_a:
            evidence_a, evidence_b = result.evidence_a, result.evidence_b
        else:
            evidence_a, evidence_b = result.evidence_b, result.evidence_a
        edge = DiscoveredConnection(
            first.id, second.id,
            candidate.preliminary_score + weights.w_evidence * result.confidence,
            result.relationship_type, result.shared_principle, evidence_a, evidence_b,
            result.confidence, model.model,
        )
        store.upsert_connection(edge)
        edges.append(edge)
    return edges


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    env = _environment()
    parser = argparse.ArgumentParser(description="Discover cross-domain ideas from notes")
    parser.add_argument("--input", required=True, help="JSON notes file, Markdown file, or Markdown directory")
    parser.add_argument("--db", help="SQLite file path or SQLite URL; defaults to DATABASE_URL")
    parser.add_argument("--top-k", type=int, default=20, help="Candidate pairs to verify (default: 20)")
    parser.add_argument("--min-confidence", type=float, default=0.60,
                        help="Minimum verifier confidence for persistence (default: 0.60)")
    parser.add_argument("--export-graph", nargs="?", const="graph.jsonld",
                        help="Export graph as JSON-LD (.jsonld) or GraphML (.graphml)")
    parser.add_argument("--model", default=env.get("GEMINI_MODEL") or DEFAULT_MODEL,
                        help=f"Gemma model (default: {DEFAULT_MODEL})")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    database = args.db
    if database and "://" not in database:
        database = "sqlite:///" + str(Path(database).resolve()).replace("\\", "/")
    database = database or _environment().get("DATABASE_URL") or DEFAULT_DATABASE_URL
    try:
        incoming = ingest_input(args.input)
        with IdeaStore(database) as store:
            count = store.upsert_notes(incoming)
            edges = discover_ideas(store, top_k=args.top_k, min_confidence=args.min_confidence,
                                   verifier=GemmaVerifier(model=args.model))
            exported = None
            if args.export_graph:
                output = Path(args.export_graph)
                if output.suffix.lower() == ".graphml":
                    exported = store.export_graphml(output)
                else:
                    exported = store.export_jsonld(output)
        print(json.dumps({"notes_upserted": count, "connections_saved": len(edges),
                          "exported_graph": str(exported) if exported else None}, indent=2))
        return 0
    except (IdeaDiscoveryError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"idea_discovery: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
