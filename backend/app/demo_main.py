"""Small, deterministic demonstration of the idea discovery pipeline.

Run from ``backend`` with ``python -m app.main``. Pass ``--real-gemini`` to
replace the clearly labeled demo verifier with configured Gemma inference.
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path

# Support both ``python -m app.main`` from backend/ and direct execution from
# the repository root (``python backend/app/main.py``).
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.idea_discovery import (
    Candidate,
    DiscoveredConnection,
    GemmaVerifier,
    IdeaStore,
    Note,
    Verification,
    Weights,
    assign_clusters,
    cosine,
    discover_ideas,
    score_candidates,
)


def _note(note_id: str, title: str, domain: str, content: str,
          cluster: int, angle_degrees: float) -> Note:
    angle = math.radians(angle_degrees)
    return Note(note_id, title, domain, content, cluster,
                [math.cos(angle), math.sin(angle)])


def sample_notes() -> list[Note]:
    """Two nearby domains with a clear analogy, plus a semantically distant domain."""
    return [
        _note("biology-local-rules", "Ant colony coordination", "Biology",
              "Each ant responds to nearby signals rather than a central commander. "
              "Simple local rules let the colony coordinate its work.", 0, -2),
        _note("biology-trail-signals", "Collective foraging", "Biology",
              "Ants leave and follow local chemical trail signals, allowing the group "
              "to adapt without a single ant directing every worker.", 0, 2),
        _note("network-local-routing", "Resilient mesh routing", "Computer networks",
              "Each router reacts to local congestion instead of a central controller. "
              "Local routing decisions let a mesh network adapt to changing traffic.", 1, 50),
        _note("network-neighbor-paths", "Distributed packet delivery", "Computer networks",
              "A node forwards packets to a suitable neighbor using current local link "
              "conditions, so the network can keep operating when routes change.", 1, 54),
        _note("astronomy-starlight", "Measuring stellar spectra", "Astronomy",
              "Astronomers split starlight into wavelengths to estimate the chemical "
              "composition and temperature of distant stars.", 2, 120),
        _note("astronomy-orbits", "Observing planetary motion", "Astronomy",
              "Repeated telescope observations reveal the positions and orbital periods "
              "of planets around a star.", 2, 124),
    ]


class DemoVerifier:
    """Deterministic stand-in that accepts one obvious analogy and rejects the rest."""

    model = "demo-rule-based (not an LLM)"

    def __init__(self):
        self.results: dict[frozenset[str], Verification] = {}

    def verify(self, a: Note, b: Note) -> Verification:
        pair = {a.id, b.id}
        if pair == {"biology-local-rules", "network-local-routing"}:
            # Evidence is copied verbatim from the corresponding source notes.
            if a.id == "biology-local-rules":
                result = Verification(
                    "structural",
                    "Both systems coordinate through local responses rather than central control.",
                    "Each ant responds to nearby signals rather than a central commander.",
                    "Each router reacts to local congestion instead of a central controller.",
                    0.96,
                )
            else:
                result = Verification(
                    "structural",
                    "Both systems coordinate through local responses rather than central control.",
                    "Each router reacts to local congestion instead of a central controller.",
                    "Each ant responds to nearby signals rather than a central commander.",
                    0.96,
                )
        else:
            result = Verification("none", "No sufficiently supported connection.", "", "", 0.08)
        self.results[frozenset(pair)] = result
        return result


class RecordingVerifier:
    """Record decisions while keeping the same verifier interface as the pipeline."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.model = delegate.model
        self.results: dict[frozenset[str], Verification] = {}

    def verify(self, a: Note, b: Note) -> Verification:
        result = self.delegate.verify(a, b)
        self.results[frozenset((a.id, b.id))] = result
        return result


def _pair_metrics(a: Note, b: Note, centroids: dict[int, list[float]],
                  notes: list[Note], weights: Weights) -> Candidate:
    semantic = cosine(a.embedding or [], b.embedding or [])
    bridge = cosine(a.embedding or [], centroids[int(b.cluster_id)]) * cosine(
        b.embedding or [], centroids[int(a.cluster_id)])

    def novelty(note: Note) -> float:
        others = [other for other in notes if other.id != note.id]
        return 1 - max((cosine(note.embedding or [], other.embedding or []) for other in others), default=0)

    variance_a = (1 - cosine(a.embedding or [], centroids[int(a.cluster_id)])) / 2
    variance_b = (1 - cosine(b.embedding or [], centroids[int(b.cluster_id)])) / 2
    novelty_score = (novelty(a) + novelty(b)) / 2
    variance = (variance_a + variance_b) / 2
    preliminary = (weights.w_s * semantic + weights.w_b * bridge
                   + weights.w_n * novelty_score + weights.w_v * variance)
    return Candidate(a, b, semantic, novelty_score, bridge, variance, preliminary)


def run_demo(*, real_gemini: bool = False, top_k: int = 20,
             min_confidence: float = 0.60) -> list[DiscoveredConnection]:
    notes = sample_notes()
    weights = Weights()
    candidates = score_candidates(notes, weights)
    candidate_pairs = {frozenset((item.note_a.id, item.note_b.id)) for item in candidates}
    centroids = assign_clusters(notes)
    delegate = GemmaVerifier() if real_gemini else DemoVerifier()
    verifier = RecordingVerifier(delegate)

    # In-memory storage keeps the demo repeatable and avoids leaving demo data in the user's DB.
    with IdeaStore("sqlite:///:memory:") as store:
        store.upsert_notes(notes)
        saved = discover_ideas(store, top_k=top_k, min_confidence=min_confidence,
                               weights=weights, verifier=verifier)

    print("Idea discovery demo")
    print(f"Verifier: {verifier.model}")
    print(f"Candidates sent to verifier: {min(top_k, len(candidates))} / {len(candidates)}")
    print(f"Saved connections: {len(saved)} (minimum confidence {min_confidence:.2f})")
    print("\nPair scores (all cross-cluster pairs):")
    print("pair | semantic | novelty | bridge | variance | preliminary | decision")
    for a, b in combinations(notes, 2):
        if a.cluster_id == b.cluster_id:
            continue
        metrics = _pair_metrics(a, b, centroids, notes, weights)
        key = frozenset((a.id, b.id))
        if key not in candidate_pairs:
            reason = "filtered: bridge <= 0.30" if metrics.bridge <= 0.30 else "filtered: semantic >= 0.70"
        elif key not in verifier.results:
            reason = "ranked below top-k"
        else:
            result = verifier.results[key]
            reason = f"{result.relationship_type}, confidence={result.confidence:.2f}"
        print(f"{a.title} <-> {b.title} | {metrics.semantic:.3f} | {metrics.novelty:.3f} | "
              f"{metrics.bridge:.3f} | {metrics.variance:.3f} | {metrics.preliminary_score:.3f} | {reason}")

    if saved:
        by_id = {note.id: note for note in notes}
        print("\nPersisted connection details:")
        for edge in saved:
            print(f"- {by_id[edge.source_id].title} <-> {by_id[edge.target_id].title}")
            print(f"  relationship_type: {edge.relationship_type}")
            print(f"  shared_principle: {edge.shared_principle}")
            print(f"  evidence_a: {edge.evidence_a}")
            print(f"  evidence_b: {edge.evidence_b}")
            print(f"  confidence: {edge.confidence:.3f}")
            print(f"  composite_score: {edge.composite_score:.3f}")
            print(f"  verified_by: {edge.verified_by}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an obvious-connection idea discovery demo")
    parser.add_argument("--real-gemini", action="store_true",
                        help="Use configured Gemma inference instead of the labeled deterministic demo verifier")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    args = parser.parse_args()
    run_demo(real_gemini=args.real_gemini, top_k=args.top_k, min_confidence=args.min_confidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
