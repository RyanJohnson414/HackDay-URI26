import { useMemo, useState } from "react";
import type { GraphResponse, GraphTopic } from "../types";

type TopicPoint = GraphTopic & { x: number; y: number; sources: number };
type TopicEdge = { id: string; source: string; target: string; score: number; title: string };
type SourcePoint = { id: string; topicId: string; x: number; y: number; label: string };

function graphLayout(graph: GraphResponse): { topics: TopicPoint[]; edges: TopicEdge[]; sources: SourcePoint[] } {
  const chunks = new Map(graph.chunks.map((chunk) => [chunk.id, chunk]));
  const docs = new Map(graph.documents.map((doc) => [doc.id, doc]));
  const topics = graph.topics.length ? graph.topics : graph.documents.map((doc) => ({
    id: `document:${doc.id}`, label: doc.title || doc.filename, keywords: doc.tags,
    chunk_ids: graph.chunks.filter((chunk) => chunk.document_id === doc.id).map((chunk) => chunk.id),
  }));
  const topicsById = new Map(topics.map((topic) => [topic.id, topic]));
  const chunkTopic = new Map<string, string>();
  topics.forEach((topic) => topic.chunk_ids.forEach((id) => chunkTopic.set(id, topic.id)));
  const sourceCounts = new Map<string, Set<string>>();
  topics.forEach((topic) => {
    const ids = new Set(topic.chunk_ids.map((id) => chunks.get(id)?.document_id).filter((id): id is string => Boolean(id)));
    sourceCounts.set(topic.id, ids);
  });
  const points: TopicPoint[] = topics.map((topic, index) => {
    const angle = -Math.PI / 2 + 2 * Math.PI * index / Math.max(topics.length, 1);
    const radius = topics.length < 2 ? 0 : Math.min(150, 75 + topics.length * 9);
    return { ...topic, x: 440 + Math.cos(angle) * radius, y: 205 + Math.sin(angle) * radius,
      sources: sourceCounts.get(topic.id)?.size || 0 };
  });
  const sources: SourcePoint[] = points.flatMap((topic) => {
    const docsForTopic = [...new Set(topic.chunk_ids.map((id) => chunks.get(id)?.document_id)
      .filter((id): id is string => Boolean(id)))];
    return docsForTopic.map((documentId, index) => {
      const angle = -Math.PI / 2 + 2 * Math.PI * index / Math.max(docsForTopic.length, 1);
      const radius = 67 + Math.min(docsForTopic.length, 8) * 2;
      const doc = docs.get(documentId);
      return { id: `${topic.id}:${documentId}`, topicId: topic.id,
        x: topic.x + Math.cos(angle) * radius, y: topic.y + Math.sin(angle) * radius,
        label: doc?.title || doc?.filename || "Source" };
    });
  });

  const edgeMap = new Map<string, TopicEdge>();
  for (const connection of graph.connections) {
    const source = chunkTopic.get(connection.source_chunk_id);
    const target = chunkTopic.get(connection.target_chunk_id);
    if (!source || !target || source === target || !topicsById.has(source) || !topicsById.has(target)) continue;
    const id = [source, target].sort().join("::");
    const existing = edgeMap.get(id);
    if (!existing || connection.similarity > existing.score) {
      edgeMap.set(id, { id, source, target, score: connection.similarity, title: connection.title });
    }
  }
  for (const relation of graph.related_topics) {
    const id = [relation.source_topic_id, relation.target_topic_id].sort().join("::");
    if (!topicsById.has(relation.source_topic_id) || !topicsById.has(relation.target_topic_id) || edgeMap.has(id)) continue;
    edgeMap.set(id, { id, source: relation.source_topic_id, target: relation.target_topic_id,
      score: relation.similarity, title: "Related topic groups" });
  }
  // Legacy or in-progress graphs may have chunks without persisted topic IDs.
  if (!edgeMap.size && graph.connections.length && points.length > 1) {
    for (const connection of graph.connections) {
      const a = chunks.get(connection.source_chunk_id);
      const b = chunks.get(connection.target_chunk_id);
      const docA = a && docs.get(a.document_id);
      const docB = b && docs.get(b.document_id);
      const source = points.find((topic) => topic.chunk_ids.some((id) => chunks.get(id)?.document_id === docA?.id));
      const target = points.find((topic) => topic.chunk_ids.some((id) => chunks.get(id)?.document_id === docB?.id));
      if (!source || !target || source.id === target.id) continue;
      const id = [source.id, target.id].sort().join("::");
      const current = edgeMap.get(id);
      if (!current || connection.similarity > current.score) {
        edgeMap.set(id, { id, source: source.id, target: target.id,
          score: connection.similarity, title: connection.title });
      }
    }
  }
  return { topics: points, edges: [...edgeMap.values()], sources };
}

export default function KnowledgeGraph({ graph, isDemo }: { graph: GraphResponse; isDemo: boolean }) {
  const layout = useMemo(() => graphLayout(graph), [graph]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<TopicEdge | null>(null);
  const selected = layout.topics.find((topic) => topic.id === selectedId) || layout.topics[0];
  const points = new Map(layout.topics.map((topic) => [topic.id, topic]));
  const selectedEdgeConnections = selectedEdge ? graph.connections.filter((connection) => {
    const source = graph.chunks.find((chunk) => chunk.id === connection.source_chunk_id)?.topic_id;
    const target = graph.chunks.find((chunk) => chunk.id === connection.target_chunk_id)?.topic_id;
    return (source === selectedEdge.source && target === selectedEdge.target)
      || (source === selectedEdge.target && target === selectedEdge.source);
  }).sort((a, b) => b.weight - a.weight) : [];
  const relationItems = selected ? graph.connections.flatMap((connection) => {
    const source = graph.chunks.find((chunk) => chunk.id === connection.source_chunk_id);
    const target = graph.chunks.find((chunk) => chunk.id === connection.target_chunk_id);
    if (!source || !target || (source.topic_id !== selected.id && target.topic_id !== selected.id)) return [];
    const sourceDoc = graph.documents.find((doc) => doc.id === source.document_id);
    const targetDoc = graph.documents.find((doc) => doc.id === target.document_id);
    return [{ id: connection.id, title: connection.title, explanation: connection.explanation,
      score: connection.similarity, weight: connection.weight,
      pair: `${sourceDoc?.title || sourceDoc?.filename || "Source"} ↔ ${targetDoc?.title || targetDoc?.filename || "Source"}` }];
  }).sort((a, b) => b.weight - a.weight).slice(0, 5) : [];

  if (!layout.topics.length) {
    return <div className="compact-graph-empty"><strong>Your graph is ready for its first source.</strong>
      <p>Add a document to see its topics and connections appear here.</p></div>;
  }

  return (
    <div className="compact-graph">
      <aside className="compact-relations">
        <div className="compact-relations-head"><span>STRONGEST RELATIONS</span><small>{selected?.label || "Select a topic"}</small></div>
        {relationItems.length ? relationItems.map((item, index) => <article className="compact-relation" key={item.id}>
          <div className="compact-relation-score">{item.score.toFixed(2)}</div>
          <div><strong>{item.title}</strong><small>{item.pair}</small><p>{item.explanation}</p></div>
          <b>{index + 1}</b>
        </article>) : <p className="compact-relations-empty">{selected ? "No verified cross-topic relations yet." : "Select a topic to inspect its strongest relations."}</p>}
        {isDemo && <span className="compact-demo">SAMPLE GRAPH</span>}
      </aside>
      <div className="compact-map-wrap">
        <svg className="compact-map" viewBox="0 0 880 410" role="img" aria-label="Compact topic relationship graph">
          <defs><filter id="compact-glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
          {layout.edges.map((edge) => {
            const a = points.get(edge.source); const b = points.get(edge.target);
            if (!a || !b) return null;
            const activeRelation = selected?.id === a.id || selected?.id === b.id;
            const x = (a.x + b.x) / 2; const y = (a.y + b.y) / 2;
            return <g key={edge.id} className="compact-edge" onClick={(event) => {
              event.stopPropagation(); setSelectedEdge(edge); setSelectedId(edge.source);
            }}>
              <path d={`M ${a.x} ${a.y} Q ${(a.x + b.x) / 2 + 8} ${(a.y + b.y) / 2 - 14} ${b.x} ${b.y}`}
              stroke="#143d27" strokeWidth={selectedEdge?.id === edge.id ? 4.5 : activeRelation ? 3.8 : 3.2} fill="none" />
              <text x={x} y={y - 10} textAnchor="middle" className="compact-edge-score">{edge.score.toFixed(2)}</text>
              <title>{edge.title}: similarity {edge.score.toFixed(2)}</title>
            </g>;
          })}
          {layout.sources.map((source) => {
            const topic = points.get(source.topicId);
            if (!topic) return null;
            const dx = source.x - topic.x; const dy = source.y - topic.y;
            const distance = Math.max(Math.hypot(dx, dy), 1);
            return <g key={source.id} className="compact-source-node">
              <line x1={topic.x + dx / distance * 39} y1={topic.y + dy / distance * 39}
                x2={source.x} y2={source.y} stroke="#143d27" strokeWidth="1.8" />
              <circle cx={source.x} cy={source.y} r="5" />
              <title>{source.label} / member of {topic.label}</title>
            </g>;
          })}
          {layout.topics.map((topic) => <g key={topic.id} className={`compact-topic${selected?.id === topic.id ? " active" : ""}`}
            onClick={() => { setSelectedId(topic.id); setSelectedEdge(null); }} role="button" tabIndex={0} aria-label={`Select ${topic.label}`}>
            <circle cx={topic.x} cy={topic.y} r={selected?.id === topic.id ? 54 : 47} className="compact-topic-ring" />
            <circle cx={topic.x} cy={topic.y} r="39" className="compact-topic-node" filter="url(#compact-glow)" />
            <text x={topic.x} y={topic.y - 2} textAnchor="middle" className="compact-topic-name">{topic.label.length > 20 ? `${topic.label.slice(0, 18)}...` : topic.label}</text>
            <text x={topic.x} y={topic.y + 17} textAnchor="middle" className="compact-topic-meta">{topic.sources} sources</text>
          </g>)}
        </svg>
        {selectedEdge && <aside className="compact-edge-popup">
          <button onClick={() => setSelectedEdge(null)} aria-label="Close relation popup">X</button>
          <span>PRIMARY TOPIC RELATION</span>
          <strong>{points.get(selectedEdge.source)?.label} / {points.get(selectedEdge.target)?.label}</strong>
          <p>Similarity {selectedEdge.score.toFixed(2)}. {selectedEdge.title}</p>
          {selectedEdgeConnections.length > 0 ? <>
            <small>CONNECTING SUBTOPICS</small>
            <div className="compact-popup-tags">{[...new Set(selectedEdgeConnections.map((item) => item.shared_concept).filter(Boolean))]
              .map((concept) => <i key={concept}>{concept}</i>)}</div>
            {selectedEdgeConnections.slice(0, 2).map((connection) => <p key={connection.id}>{connection.explanation}</p>)}
          </> : <small>Topic-group similarity; no verified passage pair is attached.</small>}
        </aside>}
        <div className="compact-map-legend"><i /> Verified connections <span>Line label = semantic similarity</span></div>
      </div>
    </div>
  );
}
