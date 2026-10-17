"""File-backed graph storage for local demonstrations; no persistent SQL database."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import BACKEND_DIR
from app.schemas.document import IngestionResponse
from app.schemas.graph import GraphResponse
from app.services.chunking import TextChunk, chunk_document
from app.services.connection_engine import ConnectionEngine
from app.services.document_parser import ParsedDocument, ParsedPage, parse_document
from app.services.embeddings import EmbeddedChunk, EmbeddingService, cosine_similarity
from app.services.gemini_client import GeminiClient
from app.services.webpage_reader import fetch_webpage

logger = logging.getLogger(__name__)
TOPIC_NAMESPACE = uuid.UUID("d8977a4f-4369-4883-bf76-6548f4af8458")
TOPIC_SIMILARITY = 0.72
RELATED_SIMILARITY = 0.58
MAX_CONNECTION_CHECKS_PER_UPLOAD = 8
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STOP_WORDS = set("""about after again against all also among an and any are as at be because been
before being between both but by can could did do does each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself may me more most
must my myself no nor not of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them this those through to too under until up very
was we were what when where which while who whom why will with would you your yours use used using
based research study studies result results shows show might one two new paper document data""".split())


class LocalDemoStore:
    """Persist source files and a JSON graph locally, while reusing the project AI pipeline."""

    def __init__(self, *, state_path: Path | None = None, source_dir: Path | None = None,
                 seed_path: Path | None = None, gemini: GeminiClient | None = None,
                 embedder: EmbeddingService | None = None,
                 connection_engine: ConnectionEngine | None = None):
        self.state_path = state_path or BACKEND_DIR / "local_demo_state.json"
        self.source_dir = source_dir or BACKEND_DIR / "local_demo_sources"
        self.seed_path = seed_path or BACKEND_DIR / "local_demo_seed.json"
        self.gemini = gemini or GeminiClient()
        self.embedder = embedder or EmbeddingService()
        self.connection_engine = connection_engine or ConnectionEngine(gemini_client=self.gemini)
        self._lock = threading.RLock()
        self._state = self._load_state()
        if not self.state_path.is_file():
            self._save()

    def _load_state(self) -> dict[str, list[dict[str, Any]]]:
        if self.state_path.is_file():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        elif self.seed_path.is_file():
            raw = json.loads(self.seed_path.read_text(encoding="utf-8"))
        else:
            raw = {}
        state = {key: list(raw.get(key, [])) for key in
                 ("documents", "chunks", "connections", "topics", "related_topics")}
        self._rebuild_topics(state)
        return state

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._state, ensure_ascii=False, separators=(",", ":")),
                             encoding="utf-8")
        temporary.replace(self.state_path)

    def graph(self) -> GraphResponse:
        with self._lock:
            documents = [{key: value for key, value in document.items()
                          if key != "source_hash"} for document in self._state["documents"]]
            topics = {topic["id"]: topic for topic in self._state["topics"]}
            chunks = []
            for chunk in self._state["chunks"]:
                topic = topics.get(chunk.get("topic_id"))
                centroid_distance = None
                if topic:
                    try:
                        centroid_distance = 1 - cosine_similarity(chunk["embedding"], topic["centroid"])
                    except ValueError:
                        pass
                chunks.append({key: value for key, value in chunk.items()
                               if key not in {"embedding", "embedding_model"}} | {
                                   "centroid_distance": centroid_distance,
                               })
            return GraphResponse.model_validate({
                "documents": documents,
                "chunks": chunks,
                "connections": self._state["connections"],
                "topics": [{key: value for key, value in topic.items() if key != "centroid"}
                           for topic in self._state["topics"]],
                "related_topics": self._state["related_topics"],
            })

    def ingest_upload(self, *, filename: str, content_type: str, data: bytes) -> IngestionResponse:
        if not data:
            raise ValueError("The uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Files must be 10 MB or smaller for the local demo")
        parsed = parse_document(data, filename=filename, content_type=content_type)
        return self._ingest(parsed, original_name=filename, original_data=data)

    def ingest_url(self, url: str) -> IngestionResponse:
        page = fetch_webpage(url)
        parsed = ParsedDocument(
            filename=page.title or page.url,
            media_type=page.document.media_type,
            pages=page.document.pages,
            source_url=page.url,
        )
        return self._ingest(parsed, original_name=page.title or page.url,
                            title_override=page.title or page.url)

    def ingest_note(self, *, title: str, domain: str, content: str,
                    tags: list[str]) -> IngestionResponse:
        filename = f"{title.strip()}.txt"[:512]
        parsed = ParsedDocument(filename, "text/plain", (ParsedPage(1, content.strip()),))
        return self._ingest(parsed, original_name=filename, title_override=title.strip(),
                            domain_override=domain.strip(), supplied_tags=tags)

    def _ingest(self, parsed: ParsedDocument, *, original_name: str,
                original_data: bytes | None = None, title_override: str | None = None,
                domain_override: str | None = None,
                supplied_tags: list[str] | None = None) -> IngestionResponse:
        with self._lock:
            document_id = str(uuid.uuid4())
            import hashlib
            source_hash = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
            if original_data is not None:
                source_hash = hashlib.sha256(original_data).hexdigest()

            stored_name = _safe_filename(original_name)
            self.source_dir.mkdir(parents=True, exist_ok=True)
            local_copy = original_data if original_data is not None else parsed.text.encode("utf-8")
            if Path(stored_name).suffix == "":
                stored_name += ".txt"
            (self.source_dir / f"{document_id[:8]}_{stored_name}").write_bytes(local_copy)

            duplicate = next((doc for doc in self._state["documents"]
                              if doc.get("source_hash") == source_hash
                              or (doc["filename"].casefold() == parsed.filename.casefold()
                                  and _stored_text_matches(
                                      [chunk for chunk in self._state["chunks"]
                                       if chunk["document_id"] == doc["id"]], parsed.text
                                  ))), None)
            if duplicate:
                return self._ingestion_response(duplicate, 0, 0, [])

            analysis = self.gemini.analyze_document(
                parsed.filename, [(page.page_number, page.text) for page in parsed.pages]
            )
            chunks = chunk_document(parsed, document_id)
            embedded = self.embedder.embed_chunks(chunks)
            if not embedded:
                raise ValueError("Document produced no text chunks")

            now = datetime.now(timezone.utc).isoformat()
            tags = list(dict.fromkeys([
                *(tag.strip() for tag in (supplied_tags or []) if tag.strip()),
                *(tag.strip() for tag in analysis.tags if tag.strip()),
            ]))[:20]
            document = {
                "id": document_id,
                "filename": parsed.filename[:512],
                "title": (title_override or analysis.title)[:512],
                "domain": (domain_override or analysis.domain)[:256],
                "tags": tags,
                "sections": [section.model_dump() for section in analysis.sections],
                "media_type": parsed.media_type,
                "source_url": parsed.source_url,
                "status": "processing",
                "error": None,
                "created_at": now,
                "chunk_count": len(embedded),
                "source_hash": source_hash,
            }
            new_chunks = [{
                "id": item.chunk.id,
                "document_id": document_id,
                "topic_id": None,
                "page_number": item.chunk.page_number,
                "text": item.chunk.text,
                "start_char": item.chunk.start_char,
                "end_char": item.chunk.end_char,
                "source_url": item.chunk.source_url,
                "embedding": list(item.vector),
                "embedding_model": item.model,
            } for item in embedded]

            docs = {row["id"]: row for row in [*self._state["documents"], document]}
            chunk_rows = [*self._state["chunks"], *new_chunks]
            corpus = [EmbeddedChunk(
                TextChunk(row["id"], row["document_id"], docs[row["document_id"]]["filename"],
                          row["page_number"], row["text"], row["start_char"], row["end_char"],
                          row.get("source_url")),
                tuple(row["embedding"]), row["embedding_model"],
            ) for row in chunk_rows]
            contexts = {row["id"]: _document_context(row) for row in docs.values()}
            warnings: list[str] = []
            edges = self.connection_engine.find_connections(
                corpus,
                focus_document_ids={document_id},
                document_contexts=contexts,
                max_candidates=MAX_CONNECTION_CHECKS_PER_UPLOAD,
                on_candidate_error=lambda error: warnings.append(f"Connection check skipped: {error}"),
            )

            connection_count = 0
            existing_pairs = {tuple(sorted((edge["source_chunk_id"], edge["target_chunk_id"]))): edge
                              for edge in self._state["connections"]}
            for edge in edges:
                pair = tuple(sorted((edge.source.chunk_id, edge.target.chunk_id)))
                saved = {
                    "id": edge.id,
                    "source_chunk_id": edge.source.chunk_id,
                    "target_chunk_id": edge.target.chunk_id,
                    "similarity": edge.similarity,
                    "strength": edge.strength,
                    "weight": edge.weight,
                    "title": edge.title,
                    "explanation": edge.explanation,
                    "shared_concept": edge.shared_concept,
                    "discipline_scope": _discipline_scope(
                        docs[edge.source.document_id]["domain"], docs[edge.target.document_id]["domain"]
                    ),
                    "semantic_distance": _semantic_distance(edge.weight),
                }
                if pair not in existing_pairs:
                    self._state["connections"].append(saved)
                    existing_pairs[pair] = saved
                connection_count += 1

            self._state["documents"].append(document)
            self._state["chunks"].extend(new_chunks)
            document["status"] = "partial" if warnings else "complete"
            document["error"] = "\n".join(warnings) or None
            topic_count, related_count = self._rebuild_topics(self._state)
            self._save()
            return self._ingestion_response(document, connection_count, related_count, warnings,
                                            topic_count=topic_count)

    def summarize_topic(self, topic_id: str) -> dict[str, Any]:
        with self._lock:
            topic = next((item for item in self._state["topics"] if item["id"] == topic_id), None)
            if topic is None:
                raise KeyError("Topic was not found")
            chunks = [chunk for chunk in self._state["chunks"] if chunk.get("topic_id") == topic_id]
            docs = {doc["id"]: doc for doc in self._state["documents"]}
            passages = [
                f"Source: {docs[chunk['document_id']]['title']}; page {chunk['page_number']}\n"
                f"{chunk['text'][:1800]}" for chunk in chunks[:20]
            ]
            summary = self.gemini.summarize_topic(topic["label"], topic["keywords"], passages)
            return {"topic_id": topic_id, "subject": topic["label"],
                    "summary": summary.summary, "key_points": summary.key_points,
                    "source_count": len({chunk["document_id"] for chunk in chunks})}

    @staticmethod
    def _ingestion_response(document: dict[str, Any], connection_count: int, related_count: int,
                            warnings: list[str], topic_count: int | None = None) -> IngestionResponse:
        from app.schemas.document import DocumentResponse

        response_document = {key: value for key, value in document.items() if key != "source_hash"}
        return IngestionResponse(
            document=DocumentResponse.model_validate(response_document),
            connection_count=connection_count,
            topic_count=topic_count or 0,
            related_topic_count=related_count,
            warnings=warnings,
        )

    @staticmethod
    def _rebuild_topics(state: dict[str, list[dict[str, Any]]]) -> tuple[int, int]:
        chunks = sorted(state["chunks"], key=lambda row: row["id"])
        if not chunks:
            state["topics"], state["related_topics"] = [], []
            return 0, 0
        parents = list(range(len(chunks)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left in range(len(chunks)):
            for right in range(left + 1, len(chunks)):
                try:
                    similar = cosine_similarity(chunks[left]["embedding"], chunks[right]["embedding"])
                except ValueError:
                    continue
                if similar >= TOPIC_SIMILARITY:
                    union(left, right)

        groups: dict[int, list[dict[str, Any]]] = {}
        for index, chunk in enumerate(chunks):
            groups.setdefault(find(index), []).append(chunk)
        term_sets = [set(_terms(chunk["text"])) for chunk in chunks]
        corpus_frequency: Counter[str] = Counter(term for terms in term_sets for term in terms)
        docs = {doc["id"]: doc for doc in state["documents"]}
        topics: list[dict[str, Any]] = []
        used_labels: set[str] = set()
        for members in groups.values():
            members.sort(key=lambda row: row["id"])
            dimension = len(members[0]["embedding"])
            centroid = [sum(row["embedding"][i] for row in members) / len(members)
                        for i in range(dimension)]
            topic_id = str(uuid.uuid5(TOPIC_NAMESPACE, ":".join(row["id"] for row in members)))
            doc_ids = {row["document_id"] for row in members}
            title = docs[next(iter(doc_ids))]["title"] if len(doc_ids) == 1 else None
            frequencies = Counter(term for row in members for term in _terms(row["text"]))
            ranked = sorted(frequencies, key=lambda word: (
                frequencies[word] * math.log((len(chunks) + 1) / (corpus_frequency[word] + 1)), word
            ), reverse=True)
            keywords = ranked[:3]
            base = (title or " · ".join(word.capitalize() for word in keywords) or "General topic")[:256]
            label = _unique_label(base, used_labels)
            topics.append({"id": topic_id, "label": label, "keywords": keywords,
                           "centroid": centroid, "chunk_ids": [row["id"] for row in members]})
            for row in members:
                row["topic_id"] = topic_id

        links = []
        for index, left in enumerate(topics):
            for right in topics[index + 1:]:
                try:
                    similarity = cosine_similarity(left["centroid"], right["centroid"])
                except ValueError:
                    continue
                if similarity >= RELATED_SIMILARITY:
                    links.append({"source_topic_id": left["id"], "target_topic_id": right["id"],
                                  "similarity": similarity})
        state["topics"], state["related_topics"] = topics, links
        return len(topics), len(links)


def _terms(text: str) -> list[str]:
    return [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text)
            if word.lower() not in STOP_WORDS]


def _unique_label(base: str, used: set[str]) -> str:
    label = base.strip() or "General topic"
    candidate, suffix = label, 2
    while candidate.casefold() in used:
        marker = f" ({suffix})"
        candidate = f"{label[:256 - len(marker)]}{marker}"
        suffix += 1
    used.add(candidate.casefold())
    return candidate


def _safe_filename(name: str) -> str:
    clean = name.replace("\\", "/").split("/")[-1].strip()
    clean = re.sub(r"[^A-Za-z0-9._() -]", "_", clean)[:180]
    return clean or "upload"


def _stored_text_matches(chunks: list[dict[str, Any]], expected: str) -> bool:
    if not chunks:
        return False
    covered = bytearray(len(expected))
    for chunk in chunks:
        start, end = chunk["start_char"], chunk["end_char"]
        if start < 0 or end > len(expected) or expected[start:end] != chunk["text"]:
            return False
        for index in range(start, end):
            if not expected[index].isspace():
                covered[index] = 1
    return all(found or character.isspace() for found, character in zip(covered, expected))


def _document_context(document: dict[str, Any]) -> str:
    sections = [f"{item.get('title', '')}: {item.get('summary', '')}"
                for item in document.get("sections", [])
                if item.get("title") or item.get("summary")]
    fields = [f"Document title: {document['title']}", f"Domain: {document['domain']}",
              f"Tags: {', '.join(document.get('tags', []))}"]
    if sections:
        fields.append("Analyzed sections: " + " | ".join(sections[:10]))
    return "\n".join(fields)


def _discipline_scope(left: str, right: str) -> str:
    a, b = left.strip().casefold(), right.strip().casefold()
    if not a or not b or "uncategorized" in {a, b}:
        return "unknown"
    return "within_discipline" if a == b else "cross_discipline"


def _semantic_distance(weight: float) -> str:
    return "local" if weight >= 0.8 else "related" if weight >= 0.6 else "distant"


_store: LocalDemoStore | None = None
_store_lock = threading.Lock()


def get_local_demo_store() -> LocalDemoStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = LocalDemoStore()
        return _store
