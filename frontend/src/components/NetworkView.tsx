import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type PointerEvent, type WheelEvent } from "react";
import { createNote, fetchGraph, ingestUrl, summarizeTopic, uploadDocument } from "../api/client";
import { emptyGraph, type DocumentRecord, type GraphChunk, type GraphConnection, type GraphResponse, type GraphTopic } from "../types";
import "../network.css";

type NodeKind = "domain" | "topic" | "document" | "chunk";
type ViewNode = {
  id: string;
  kind: NodeKind;
  label: string;
  domain: string;
  x: number;
  y: number;
  r: number;
  topic?: GraphTopic;
  document?: GraphDocument;
  chunk?: GraphChunk;
  chunkIds: string[];
};
type ViewLink = {
  id: string;
  source: string;
  target: string;
  kind: "hierarchy" | "local" | "bridge" | "similar";
  title?: string;
  weight?: number;
  crossTopic?: boolean;
  lod?: "document" | "chunk";
  relation?: TopicBridge;
};
type TopicBridge = {
  sourceTopic: GraphTopic;
  targetTopic: GraphTopic;
  similarity: number;
  weight: number;
  connections: GraphConnection[];
};
type Layout = { nodes: ViewNode[]; links: ViewLink[] };
type Transform = { x: number; y: number; scale: number };
type GraphDocument = DocumentRecord;

function uniqueTopicLabel(label: string, usedLabels: Set<string>): string {
  const base = label.trim() || "Untitled topic";
  let candidate = base;
  let suffix = 2;
  while (usedLabels.has(candidate.toLowerCase())) {
    candidate = `${base} (${suffix})`;
    suffix += 1;
  }
  usedLabels.add(candidate.toLowerCase());
  return candidate;
}

const PALETTE = {
  background: "#fffedb",
  cream: "#fffedb",
  local: "#143d27",
  bridge: "#971111",
  muted: "#46666d",
};
const WIDTH = 1600;
const HEIGHT = 1000;

function buildLayout(graph: GraphResponse): Layout {
  const documents = new Map(graph.documents.map((document) => [document.id, document]));
  const chunks = new Map(graph.chunks.map((chunk) => [chunk.id, chunk]));
  const usedTopicLabels = new Set<string>();
  const rawTopics: GraphTopic[] = graph.topics.length ? [...graph.topics] : graph.documents.map((document) => ({
    id: `topic:${document.id}`, label: document.title || document.filename,
    keywords: document.tags, chunk_ids: graph.chunks.filter((chunk) => chunk.document_id === document.id).map((chunk) => chunk.id),
  }));
  const topics = rawTopics.map((topic) => ({
    ...topic,
    label: uniqueTopicLabel(topic.label, usedTopicLabels),
  }));

  const topicsByDomain = new Map<string, GraphTopic[]>();
  const topicDomain = new Map<string, string>();
  for (const topic of topics) {
    const counts = new Map<string, number>();
    for (const id of topic.chunk_ids) {
      const doc = documents.get(chunks.get(id)?.document_id || "");
      if (doc) counts.set(doc.domain || "Uncategorized", (counts.get(doc.domain || "Uncategorized") || 0) + 1);
    }
    const domain = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "Uncategorized";
    topicDomain.set(topic.id, domain);
    topicsByDomain.set(domain, [...(topicsByDomain.get(domain) || []), topic]);
  }

  const docTopic = new Map<string, { topic: GraphTopic; count: number }>();
  for (const topic of topics) {
    const counts = new Map<string, number>();
    for (const id of topic.chunk_ids) {
      const docId = chunks.get(id)?.document_id;
      if (docId) counts.set(docId, (counts.get(docId) || 0) + 1);
    }
    for (const [docId, count] of counts) {
      if (count > (docTopic.get(docId)?.count || 0)) docTopic.set(docId, { topic, count });
    }
  }

  // Keep files without persisted topic assignments visible as distinct, named nodes.
  for (const document of graph.documents) {
    if (docTopic.has(document.id)) continue;
    const domain = document.domain || "Uncategorized";
    const topic: GraphTopic = {
      id: `topic:unassigned:${document.id}`,
      label: uniqueTopicLabel(document.title || document.filename, usedTopicLabels),
      keywords: document.tags,
      chunk_ids: graph.chunks.filter((chunk) => chunk.document_id === document.id).map((chunk) => chunk.id),
    };
    topics.push(topic);
    topicDomain.set(topic.id, domain);
    topicsByDomain.set(domain, [...(topicsByDomain.get(domain) || []), topic]);
    docTopic.set(document.id, { topic, count: 0 });
  }

  const domainNames = [...new Set([
    ...graph.documents.map((document) => document.domain || "Uncategorized"),
    ...topicDomain.values(),
  ])].sort();
  const centerX = WIDTH / 2;
  const centerY = HEIGHT / 2;
  const domainRadius = domainNames.length <= 1 ? 0 : Math.min(340, 115 + domainNames.length * 34);
  const domainPositions = new Map<string, { x: number; y: number }>();
  domainNames.forEach((domain, index) => {
    const angle = -Math.PI / 2 + 2 * Math.PI * index / Math.max(domainNames.length, 1);
    domainPositions.set(domain, {
      x: centerX + Math.cos(angle) * domainRadius,
      y: centerY + Math.sin(angle) * domainRadius,
    });
  });

  const nodes: ViewNode[] = [];
  const links: ViewLink[] = [];
  const topicPositions = new Map<string, { x: number; y: number }>();
  for (const domain of domainNames) {
    const position = domainPositions.get(domain)!;
    const domainDocs = graph.documents.filter((document) => (document.domain || "Uncategorized") === domain);
    const domainTopics = topicsByDomain.get(domain) || [];
    nodes.push({
      id: `domain:${domain}`, kind: "domain", label: domain, domain,
      x: position.x, y: position.y, r: 26, chunkIds: domainDocs.flatMap((document) =>
        graph.chunks.filter((chunk) => chunk.document_id === document.id).map((chunk) => chunk.id)),
    });
    // Outer orbital rings are decorative marks around domain hubs.
    domainTopics.forEach((topic, topicIndex) => {
      const angle = -Math.PI / 2 + 2 * Math.PI * topicIndex / Math.max(domainTopics.length, 1);
      const radius = domainTopics.length === 1 ? 66 : Math.min(112, 52 + domainTopics.length * 9);
      const point = { x: position.x + Math.cos(angle) * radius, y: position.y + Math.sin(angle) * radius };
      topicPositions.set(topic.id, point);
      nodes.push({ id: topic.id, kind: "topic", label: topic.label, domain,
        x: point.x, y: point.y, r: 15, topic, chunkIds: topic.chunk_ids });
      links.push({ id: `domain-topic:${domain}:${topic.id}`, source: `domain:${domain}`,
        target: topic.id, kind: "hierarchy" });
    });
  }

  const docsByTopic = new Map<string, GraphDocument[]>();
  for (const document of graph.documents) {
    const parent = docTopic.get(document.id)?.topic;
    if (parent) docsByTopic.set(parent.id, [...(docsByTopic.get(parent.id) || []), document]);
  }
  for (const [topicId, members] of docsByTopic) {
    const center = topicPositions.get(topicId);
    if (!center) continue;
    members.forEach((document, docIndex) => {
      const angle = -Math.PI / 2 + 2 * Math.PI * docIndex / Math.max(members.length, 1);
      const radius = members.length === 1 ? 56 : 62 + Math.min(26, members.length * 3);
      const point = { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius };
      const chunkIds = graph.chunks.filter((chunk) => chunk.document_id === document.id).map((chunk) => chunk.id);
      nodes.push({ id: `document:${document.id}`, kind: "document",
        label: document.title || document.filename, domain: document.domain, x: point.x, y: point.y,
        r: 7, document, chunkIds });
      links.push({ id: `topic-document:${topicId}:${document.id}`, source: topicId,
        target: `document:${document.id}`, kind: "hierarchy", lod: "document" });
      const documentChunks = graph.chunks.filter((chunk) => chunk.document_id === document.id);
      documentChunks.forEach((chunk, chunkIndex) => {
        const chunkAngle = -Math.PI / 2 + 2 * Math.PI * chunkIndex / Math.max(documentChunks.length, 1);
        const chunkRadius = documentChunks.length === 1 ? 17 : 19 + Math.min(10, documentChunks.length * 1.5);
        const chunkPoint = { x: point.x + Math.cos(chunkAngle) * chunkRadius,
          y: point.y + Math.sin(chunkAngle) * chunkRadius };
        nodes.push({ id: `chunk:${chunk.id}`, kind: "chunk", label: `Page ${chunk.page_number}`,
          domain: document.domain, x: chunkPoint.x, y: chunkPoint.y, r: 3.4,
          document, chunk, chunkIds: [chunk.id] });
        links.push({ id: `document-chunk:${document.id}:${chunk.id}`, source: `document:${document.id}`,
          target: `chunk:${chunk.id}`, kind: "hierarchy", lod: "chunk" });
      });
    });
  }

  const topicById = new Map(topics.map((topic) => [topic.id, topic]));
  const topicForChunk = (chunkId: string): GraphTopic | undefined => {
    const chunk = chunks.get(chunkId);
    return topicById.get(chunk?.topic_id || "") || docTopic.get(chunk?.document_id || "")?.topic;
  };
  const bridgeGroups = new Map<string, { source: GraphTopic; target: GraphTopic;
    connections: GraphConnection[]; topicSimilarity?: number }>();
  for (const connection of graph.connections) {
    const left = topicForChunk(connection.source_chunk_id);
    const right = topicForChunk(connection.target_chunk_id);
    if (!left || !right || left.id === right.id) continue;
    const [source, target] = [left, right].sort((a, b) => a.id.localeCompare(b.id));
    const key = `${source.id}::${target.id}`;
    const group = bridgeGroups.get(key) || { source, target, connections: [] };
    group.connections.push(connection);
    bridgeGroups.set(key, group);
  }
  for (const relation of graph.related_topics) {
    const left = topicById.get(relation.source_topic_id);
    const right = topicById.get(relation.target_topic_id);
    if (!left || !right || left.id === right.id) continue;
    const [source, target] = [left, right].sort((a, b) => a.id.localeCompare(b.id));
    const key = `${source.id}::${target.id}`;
    const group = bridgeGroups.get(key) || { source, target, connections: [] };
    group.topicSimilarity = Math.max(group.topicSimilarity || 0, relation.similarity);
    bridgeGroups.set(key, group);
  }
  for (const [key, group] of bridgeGroups) {
    if (!topicPositions.has(group.source.id) || !topicPositions.has(group.target.id)) continue;
    const count = group.connections.length;
    const similarity = count
      ? group.connections.reduce((sum, item) => sum + item.similarity, 0) / count
      : group.topicSimilarity || 0;
    const weight = count
      ? Math.max(...group.connections.map((item) => item.weight), group.topicSimilarity || 0)
      : group.topicSimilarity || 0;
    const crossDomain = group.connections.some((item) => item.discipline_scope === "cross_discipline")
      || topicDomain.get(group.source.id) !== topicDomain.get(group.target.id);
    links.push({ id: `topic-bridge:${key}`, source: group.source.id, target: group.target.id,
      kind: crossDomain ? "bridge" : "local", crossTopic: true, weight,
      title: `${group.source.label} / ${group.target.label}`, relation: {
        sourceTopic: group.source, targetTopic: group.target, similarity, weight,
        connections: group.connections.sort((a, b) => b.weight - a.weight),
      } });
  }
  return { nodes, links };
}

function nodeEdgeRadius(node: ViewNode, transformScale: number): number {
  if (node.kind === "domain") return 26;
  if (node.kind === "topic") return 22;
  if (node.kind === "document") return transformScale < 0.48 ? 3.2 : transformScale < 0.8 ? 4.5 : 6.2;
  return transformScale < 0.5 ? 2.2 : transformScale < 0.9 ? 3 : 4.2;
}

function curvedPath(source: ViewNode, target: ViewNode, offset = 48, scale = 1): string {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const length = Math.max(Math.hypot(dx, dy), 1);
  const middleX = (source.x + target.x) / 2 - dy / length * offset;
  const middleY = (source.y + target.y) / 2 + dx / length * offset;
  // Trim the quadratic at both ends so bridge strokes meet the node outlines.
  const sourceRadius = nodeEdgeRadius(source, scale);
  const targetRadius = nodeEdgeRadius(target, scale);
  const sourceTangentLength = Math.max(Math.hypot(middleX - source.x, middleY - source.y), 1);
  const targetTangentLength = Math.max(Math.hypot(target.x - middleX, target.y - middleY), 1);
  const startX = source.x + (middleX - source.x) / sourceTangentLength * sourceRadius;
  const startY = source.y + (middleY - source.y) / sourceTangentLength * sourceRadius;
  const endX = target.x - (target.x - middleX) / targetTangentLength * targetRadius;
  const endY = target.y - (target.y - middleY) / targetTangentLength * targetRadius;
  return `M ${startX} ${startY} Q ${middleX} ${middleY} ${endX} ${endY}`;
}

function straightPath(source: ViewNode, target: ViewNode, scale: number, inset = 0): string {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const length = Math.max(Math.hypot(dx, dy), 1);
  const sourceRadius = nodeEdgeRadius(source, scale) + inset;
  const targetRadius = nodeEdgeRadius(target, scale) + inset;
  return `M ${source.x + dx / length * sourceRadius} ${source.y + dy / length * sourceRadius} L ${target.x - dx / length * targetRadius} ${target.y - dy / length * targetRadius}`;
}

function strengthColor(strength: number, amount: number): string {
  const pale = [225, 145, 139];
  const strong = [151, 17, 17];
  return `#${pale.map((start, index) => Math.round(start + (strong[index] - start) * strength * amount)
    .toString(16).padStart(2, "0")).join("")}`;
}

export default function NetworkView() {
  const embedded = new URLSearchParams(window.location.search).has("embedded");
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem("hauri-theme") === "dark");
  const [graph, setGraph] = useState<GraphResponse>(emptyGraph);
  const [backendConnected, setBackendConnected] = useState(false);
  const [bridgesVisible, setBridgesVisible] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedRelation, setSelectedRelation] = useState<TopicBridge | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, scale: 0.82 });
  const [modalOpen, setModalOpen] = useState(false);
  const [sourceMode, setSourceMode] = useState<"file" | "url" | "note">("file");
  const [sourceError, setSourceError] = useState("");
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState("");
  const [topicSummary, setTopicSummary] = useState<{ topicId: string; text: string; points: string[]; sources: number } | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState("");
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  const refresh = useCallback(async () => {
    try {
      const result = await fetchGraph();
      if (result.documents.length || result.topics.length || result.chunks.length) {
        setGraph(result);
        setBackendConnected(true);
        setNotice(result.documents.length > 0 && result.documents.every((document) =>
          document.tags.includes("starter-example"))
          ? "Starter example data is loaded from the local demo files. Add a source to extend it."
          : "");
      } else {
        setGraph(result);
        setBackendConnected(true);
        setNotice("No sources are stored yet. Upload a file or add a note to build your graph.");
      }
    } catch {
      setGraph(emptyGraph);
      setBackendConnected(false);
      setNotice("Local demo service is offline. Start the backend to load your graph files.");
    }
  }, []);

  useEffect(() => {
    document.body.classList.add("network-mode");
    void refresh();
    return () => document.body.classList.remove("network-mode", "network-dark-mode");
  }, [refresh]);

  useEffect(() => {
    if (backendConnected) return;
    const retry = window.setInterval(() => { void refresh(); }, 3000);
    return () => window.clearInterval(retry);
  }, [backendConnected, refresh]);

  useEffect(() => {
    document.body.classList.toggle("network-dark-mode", darkMode);
    const syncTheme = (event: StorageEvent) => {
      if (event.key === "hauri-theme") setDarkMode(event.newValue === "dark");
    };
    window.addEventListener("storage", syncTheme);
    return () => window.removeEventListener("storage", syncTheme);
  }, [darkMode]);

  const layout = useMemo(() => buildLayout(graph), [graph]);
  useEffect(() => {
    if (!embedded) return;
    const degrees = new Map<string, number>();
    for (const link of layout.links) {
      if (!link.crossTopic) continue;
      degrees.set(link.source, (degrees.get(link.source) || 0) + 1);
      degrees.set(link.target, (degrees.get(link.target) || 0) + 1);
    }
    const centralTopic = layout.nodes
      .filter((node) => node.kind === "topic")
      .sort((left, right) => (degrees.get(right.id) || 0) - (degrees.get(left.id) || 0)
        || right.chunkIds.length - left.chunkIds.length)[0];
    if (!centralTopic) return;
    const scale = 1.35;
    setTransform({ scale, x: WIDTH / 2 - centralTopic.x * scale, y: HEIGHT / 2 - centralTopic.y * scale });
  }, [embedded, layout]);
  const nodeById = useMemo(() => new Map(layout.nodes.map((node) => [node.id, node])), [layout.nodes]);
  const focusId = hoveredId || selectedId;
  const selectedNode = focusId ? nodeById.get(focusId) : undefined;
  // Generate the optional AI description when a topic is explicitly selected.
  // The inspector also has a source-grounded fallback, so it remains useful if
  // the model is unavailable or the topic has no verified connections.
  const selectedTopic = selectedId ? nodeById.get(selectedId) : undefined;
  const summaryTopicId = selectedTopic?.kind === "topic" ? selectedTopic.id : null;

  useEffect(() => {
    if (embedded || !summaryTopicId) return;
    let active = true;
    setTopicSummary(null);
    setSummaryError("");
    setSummaryLoading(true);
    summarizeTopic(summaryTopicId)
      .then((result) => {
        if (active) setTopicSummary({ topicId: result.topic_id, text: result.summary,
          points: result.key_points, sources: result.source_count });
      })
      .catch((error: unknown) => {
        if (active) setSummaryError(error instanceof Error ? error.message : "Could not summarize this topic.");
      })
      .finally(() => { if (active) setSummaryLoading(false); });
    return () => { active = false; };
  }, [embedded, summaryTopicId]);
  const visibleLinks = layout.links.filter((link) => {
    if (link.kind === "hierarchy" && link.lod === "document" && transform.scale < 0.92) return false;
    if (link.kind === "hierarchy" && link.lod === "chunk" && transform.scale < 1.18) return false;
    if (!bridgesVisible && link.crossTopic) return false;
    if (link.lod === "document" && transform.scale < 0.92) return false;
    if (link.lod === "chunk" && transform.scale < 1.18) return false;
    return true;
  });
  const neighbors = useMemo(() => {
    const found = new Set<string>();
    if (!focusId) return found;
    found.add(focusId);
    for (const link of layout.links) {
      if (link.source === focusId) found.add(link.target);
      if (link.target === focusId) found.add(link.source);
    }
    return found;
  }, [focusId, layout.links]);
  const documents = new Map(graph.documents.map((item) => [item.id, item]));
  const chunks = new Map(graph.chunks.map((item) => [item.id, item]));

  const zoomAt = (factor: number, px = WIDTH / 2, py = HEIGHT / 2) => {
    setTransform((current) => {
      const scale = Math.min(3.4, Math.max(0.24, current.scale * factor));
      const ratio = scale / current.scale;
      return { scale, x: px - (px - current.x) * ratio, y: py - (py - current.y) * ratio };
    });
  };
  function handleWheel(event: WheelEvent<SVGSVGElement>) {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const px = (event.clientX - rect.left) * WIDTH / rect.width;
    const py = (event.clientY - rect.top) * HEIGHT / rect.height;
    zoomAt(event.deltaY < 0 ? 1.12 : 0.89, px, py);
  }
  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    if ((event.target as Element).closest("[data-network-node], [data-network-relation]")) return;
    drag.current = { x: event.clientX, y: event.clientY, tx: transform.x, ty: transform.y };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!drag.current) return;
    const rect = event.currentTarget.getBoundingClientRect();
    setTransform((current) => ({ ...current,
      x: drag.current!.tx + (event.clientX - drag.current!.x) * WIDTH / rect.width,
      y: drag.current!.ty + (event.clientY - drag.current!.y) * HEIGHT / rect.height,
    }));
  }

  async function submitSource(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setSaving(true);
    setSourceError("");
    setNotice("");
    try {
      const result = sourceMode === "file"
        ? await uploadDocument(form.get("file") as File)
        : sourceMode === "url"
          ? await ingestUrl(String(form.get("url") || "").trim())
          : await createNote({
            title: String(form.get("title") || "").trim(),
            domain: String(form.get("domain") || "").trim(),
            content: String(form.get("content") || "").trim(),
            tags: String(form.get("tags") || "").split(",").map((tag) => tag.trim()).filter(Boolean),
          });
      setModalOpen(false);
      await refresh();
      setNotice(`Added "${result.document.title}" to the local demo graph.`);
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : "Could not add source.");
    } finally {
      setSaving(false);
    }
  }

  const connectedBridges = selectedNode?.kind === "document"
    ? graph.connections.filter((connection) => {
      const a = chunks.get(connection.source_chunk_id)?.document_id;
      const b = chunks.get(connection.target_chunk_id)?.document_id;
      return a === selectedNode.document?.id || b === selectedNode.document?.id;
    })
    : selectedNode?.kind === "chunk"
      ? graph.connections.filter((connection) => connection.source_chunk_id === selectedNode.chunk?.id
        || connection.target_chunk_id === selectedNode.chunk?.id)
      : selectedNode?.kind === "topic"
      ? graph.connections.filter((connection) => selectedNode.chunkIds.includes(connection.source_chunk_id)
        || selectedNode.chunkIds.includes(connection.target_chunk_id))
      : selectedNode?.kind === "domain"
        ? graph.connections.filter((connection) => {
          const left = chunks.get(connection.source_chunk_id);
          const right = chunks.get(connection.target_chunk_id);
          return documents.get(left?.document_id || "")?.domain === selectedNode.domain
            || documents.get(right?.document_id || "")?.domain === selectedNode.domain;
        })
        : [];
  const relatedChunks = selectedNode?.kind === "document"
    ? graph.chunks.filter((chunk) => chunk.document_id === selectedNode.document?.id)
    : selectedNode?.kind === "topic"
      ? graph.chunks.filter((chunk) => selectedNode.chunkIds.includes(chunk.id))
      : graph.chunks.filter((chunk) => selectedNode?.chunkIds.includes(chunk.id));
  const topicDescription = selectedNode?.kind === "topic" ? (() => {
    const sourceIds = new Set(relatedChunks.map((chunk) => chunk.document_id));
    const sourceDocs = [...sourceIds].map((id) => documents.get(id)).filter((doc): doc is DocumentRecord => Boolean(doc));
    const sectionDescriptions = sourceDocs.flatMap((doc) => doc.sections
      .map((section) => section.summary.trim()).filter(Boolean));
    if (sectionDescriptions.length) return sectionDescriptions.slice(0, 2).join(" ");
    const passage = relatedChunks.map((chunk) => chunk.text.replace(/\s+/g, " ").trim())
      .filter(Boolean).slice(0, 2).join(" ");
    if (passage) return passage.length > 520 ? `${passage.slice(0, 517).trimEnd()}...` : passage;
    const keywords = selectedNode.topic?.keywords || [];
    return keywords.length
      ? `${selectedNode.label} is a topic represented by source material about ${keywords.slice(0, 6).join(", ")}.`
      : `${selectedNode.label} is a topic identified in the source material. Add more source text to expand its description.`;
  })() : "";
  const activeTopicSummary = selectedNode?.kind === "topic" && topicSummary?.topicId === selectedNode.id
    ? topicSummary : null;
  const centroidDistance = relatedChunks.length
    ? relatedChunks.map((chunk) => chunk.centroid_distance).filter((value): value is number => value !== null)
      .reduce((sum, value, _, all) => sum + value / all.length, 0)
    : null;

  return (
    <main className={`network-shell${embedded ? " embedded" : ""}`}>
      <header className="network-topbar">
        <div className="network-brand"><span className="network-mark">C</span><span>CONSILIENCE <i>/</i> NETWORK</span></div>
        <div className="network-actions">
          <span className="network-source-status"><b />{backendConnected ? "SQLITE GRAPH" : "BACKEND OFFLINE"}</span>
          <button className={`network-toggle${bridgesVisible ? " active" : ""}`} onClick={() => setBridgesVisible((visible) => !visible)}>
            <span className="toggle-icon">||</span> Topic bridges <span className="toggle-state">{bridgesVisible ? "ON" : "OFF"}</span>
          </button>
          <button className="network-toggle" onClick={() => setDarkMode((mode) => {
            const next = !mode;
            localStorage.setItem("hauri-theme", next ? "dark" : "light");
            return next;
          })}>{darkMode ? "Light mode" : "Dark mode"}</button>
          <a className="network-back" href="/">Dashboard -&gt;</a>
        </div>
      </header>

      {notice && <div className="network-notice">{notice}</div>}

      <svg className="network-svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Interactive hierarchical knowledge network"
        onWheel={handleWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove}
        onClick={() => setSelectedRelation(null)}
        onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
        <defs>
          <filter id="hub-glow" x="-200%" y="-200%" width="500%" height="500%">
            <feGaussianBlur stdDeviation="9" result="blur" />
            <feFlood floodColor={PALETTE.cream} floodOpacity=".28" />
            <feComposite in2="blur" operator="in" />
            <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="bridge-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <rect width={WIDTH} height={HEIGHT} fill={PALETTE.background} />
        <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.scale})`}>
          {transform.scale >= 0.92 && layout.nodes.filter((node) => node.kind === "topic").map((node) => (
            <g key={`orbit:${node.id}`} className="planetary-rings">
              <circle cx={node.x} cy={node.y} r="29" />
              <circle cx={node.x} cy={node.y} r="43" />
              <circle cx={node.x} cy={node.y} r="58" />
            </g>
          ))}
          {visibleLinks.map((link) => {
            const source = nodeById.get(link.source);
            const target = nodeById.get(link.target);
            if (!source || !target) return null;
            const focused = !focusId || neighbors.has(source.id) && neighbors.has(target.id);
            const opacity = focusId ? focused ? 0.95 : 0.1 : 0.78;
            const isBridge = link.kind === "bridge";
            const color = link.kind === "hierarchy" || link.kind === "local" ? (darkMode ? PALETTE.cream : PALETTE.local) :
              link.kind === "bridge" ? PALETTE.bridge : PALETTE.muted;
            const isGreenLink = link.kind === "hierarchy" || link.kind === "local";
            const d = isBridge ? curvedPath(source, target, 58, transform.scale)
              : straightPath(source, target, transform.scale, isGreenLink ? 6 : 0);
            const score = link.relation?.similarity ?? link.weight;
            const scoreX = (source.x + target.x) / 2;
            const scoreY = (source.y + target.y) / 2 - (isBridge ? 28 : 10);
            const strength = Math.max(0, Math.min(1, link.relation?.similarity ?? link.weight ?? 0));
            const relationshipWidth = 0.65 + strength * 1.6;
            const gradientId = `bridge-gradient-${link.id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
            return <g key={link.id} opacity={opacity} data-network-relation={link.relation ? "true" : undefined}
              className={link.relation ? "topic-bridge-link" : undefined}
              onClick={(event) => {
                if (!link.relation) return;
                event.stopPropagation();
                setSelectedRelation(link.relation);
              }}>
              {isBridge && <defs><linearGradient id={gradientId} x1={source.x} y1={source.y} x2={target.x} y2={target.y} gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor={strengthColor(strength, 0.68)} stopOpacity={0.68 + strength * 0.2} />
                <stop offset="100%" stopColor={strengthColor(strength, 1)} stopOpacity={0.78 + strength * 0.2} />
              </linearGradient></defs>}
              <path d={d} className={isBridge ? "network-bridge-line" : undefined}
                stroke={isBridge ? `url(#${gradientId})` : color}
                strokeWidth={isGreenLink || isBridge ? relationshipWidth : 2.5}
                opacity={link.kind === "hierarchy" ? 0.9 : 1} fill="none"
                filter={isBridge && focused ? "url(#bridge-glow)" : undefined}>
                <title>{link.relation
                  ? `${link.title}: ${link.relation.connections.length} verified connection${link.relation.connections.length === 1 ? "" : "s"}`
                  : link.title || (link.kind === "hierarchy" ? "Topic hierarchy" : "Local relationship")}</title>
              </path>
              {link.kind !== "hierarchy" && score !== undefined && <text x={scoreX} y={scoreY}
                className="network-link-score" textAnchor="middle">{score.toFixed(2)}</text>}
            </g>;
          })}
          {layout.nodes.filter((node) =>
            !(node.kind === "document" && transform.scale < 0.92)
            && !(node.kind === "chunk" && transform.scale < 1.18)
          ).map((node) => {
            const focused = !focusId || neighbors.has(node.id);
            const opacity = focusId ? focused ? 1 : 0.16 : 1;
            const scaleRadius = node.kind === "document"
              ? transform.scale < 0.48 ? 3.2 : transform.scale < 0.8 ? 4.5 : 6.2
              : node.kind === "chunk" ? transform.scale < 0.5 ? 2.2 : transform.scale < 0.9 ? 3 : 4.2
              : node.r;
            const isSelected = selectedId === node.id;
            return <g key={node.id} data-network-node="true" className={`network-node node-${node.kind}${isSelected ? " selected" : ""}`}
              transform={`translate(${node.x} ${node.y})`} opacity={opacity}
              onMouseEnter={() => setHoveredId(node.id)} onMouseLeave={() => setHoveredId(null)}
              onClick={(event) => {
                event.stopPropagation();
                setSelectedId(node.id);
                setSelectedRelation(null);
                if (node.kind === "topic") {
                  const scale = Math.max(transform.scale, 1.45);
                  setTransform({ scale, x: WIDTH / 2 - node.x * scale, y: HEIGHT / 2 - node.y * scale });
                }
              }}>
              {node.kind === "topic" && <>
                <circle r="22" className="topic-outer-ring" />
                <circle r="15" className="topic-hub" />
              </>}
              {node.kind === "document" && <circle r={scaleRadius} className="document-node" />}
              {node.kind === "chunk" && <circle r={scaleRadius} className="chunk-node" />}
              {(node.kind === "domain" || node.kind === "topic" || (node.kind === "document" && transform.scale >= 1.08)) && (
                <text y={node.kind === "domain" ? 8 : node.kind === "topic" ? 32 : 17}
                  className={`node-label label-${node.kind}`} textAnchor="middle">
                  {node.label.length > 29 ? `${node.label.slice(0, 27)}...` : node.label}
                </text>
              )}
              {node.kind === "document" && transform.scale >= 1.18 && (
                <text y="30" className="node-sublabel" textAnchor="middle">
                  {node.document?.tags.slice(0, 2).join(" / ")}
                </text>
              )}
              <title>{node.label}{node.kind === "document" ? ` / ${node.domain}` : node.kind === "chunk" ? ` / ${node.document?.title || node.document?.filename}, page ${node.chunk?.page_number}` : ""}</title>
            </g>;
          })}
        </g>
      </svg>

      {selectedRelation && <section className="topic-relation-popup" role="dialog" aria-label="Topic relationship details">
        <button className="topic-relation-close" onClick={() => setSelectedRelation(null)} aria-label="Close relationship details">X</button>
        <span className="relation-popup-eyebrow">PRIMARY TOPIC RELATION</span>
        <h2>{selectedRelation.sourceTopic.label} <i>to</i> {selectedRelation.targetTopic.label}</h2>
        <div className="relation-popup-values">
          <div><small>SIMILARITY</small><strong>{selectedRelation.similarity.toFixed(2)}</strong></div>
          <div><small>RELATION WEIGHT</small><strong>{selectedRelation.weight.toFixed(2)}</strong></div>
          <div><small>VERIFIED LINKS</small><strong>{selectedRelation.connections.length}</strong></div>
        </div>
        {selectedRelation.connections.length > 0 ? <>
          <div className="relation-subtopic-group"><span>CONNECTING SUBTOPICS</span>
            <div className="inspector-tags">
              {[...new Set(selectedRelation.connections.map((item) => item.shared_concept).filter(Boolean))]
                .map((concept) => <i key={concept}>{concept}</i>)}
            </div>
          </div>
          <div className="relation-subtopic-columns">
            <div><small>{selectedRelation.sourceTopic.label}</small><div className="inspector-tags">
              {selectedRelation.sourceTopic.keywords.map((word) => <i key={word}>{word}</i>)}
            </div></div>
            <div><small>{selectedRelation.targetTopic.label}</small><div className="inspector-tags">
              {selectedRelation.targetTopic.keywords.map((word) => <i key={word}>{word}</i>)}
            </div></div>
          </div>
          <div className="relation-evidence-list"><span>PASSAGES THAT SUPPORT THIS LINK</span>
            {selectedRelation.connections.slice(0, 4).map((connection) => {
              const left = chunks.get(connection.source_chunk_id);
              const right = chunks.get(connection.target_chunk_id);
              const leftDoc = left ? documents.get(left.document_id) : undefined;
              const rightDoc = right ? documents.get(right.document_id) : undefined;
              return <article key={connection.id}>
                <strong>{connection.title}</strong>
                <small>{leftDoc?.title || leftDoc?.filename} p.{left?.page_number} / {rightDoc?.title || rightDoc?.filename} p.{right?.page_number}</small>
                <p>{connection.explanation}</p>
              </article>;
            })}
          </div>
        </> : <div className="relation-subtopic-group"><span>TOPIC KEYWORDS</span>
          <div className="inspector-tags">
            {[...new Set([...selectedRelation.sourceTopic.keywords, ...selectedRelation.targetTopic.keywords])]
              .map((word) => <i key={word}>{word}</i>)}
          </div><p className="relation-keyword-note">This link comes from topic-centroid similarity; no passage pair has been verified yet.</p>
        </div>}
      </section>}

      <div className="network-legend">
        <span><i className="legend-line local-line" />LOCAL RELATIONSHIP</span>
        <span><i className="legend-line bridge-line" />CROSS-DOMAIN BRIDGE</span>
        <span><i className="legend-line similar-line" />SIMILAR TOPIC</span>
      </div>

      <div className="network-zoom-controls">
        <button onClick={() => zoomAt(1.18)} aria-label="Zoom in">+</button>
        <button onClick={() => zoomAt(0.84)} aria-label="Zoom out">-</button>
        <button onClick={() => setTransform({ x: 0, y: 0, scale: 0.82 })} aria-label="Reset view">R</button>
      </div>

      <button className="add-source-button" onClick={() => { setSourceError(""); setModalOpen(true); }}><span>+</span> ADD SOURCE</button>

      <aside className={`inspector${selectedNode ? " open" : ""}`}>
        <div className="inspector-top">
          <div><span>NETWORK INSPECTOR</span><small>{selectedNode ? selectedNode.kind.toUpperCase() : "HOVER OR SELECT A NODE"}</small></div>
          {selectedNode && <button onClick={() => { setSelectedId(null); setHoveredId(null); }} aria-label="Close inspector">X</button>}
        </div>
        {selectedNode && <div className="inspector-content">
          <div className="inspector-title-line"><span className={`inspector-node-dot ${selectedNode.kind}`} />
            <h2>{selectedNode.label}</h2></div>
          <dl className="metadata-grid">
            <div><dt>DOMAIN</dt><dd>{selectedNode.kind === "domain" ? selectedNode.label : selectedNode.domain}</dd></div>
            <div><dt>PASSAGES</dt><dd>{selectedNode.chunkIds.length}</dd></div>
            {centroidDistance !== null && <div><dt>CLUSTER DISTANCE</dt><dd>{centroidDistance.toFixed(3)}</dd></div>}
            {selectedNode.document && <div><dt>SOURCE STATUS</dt><dd>{selectedNode.document.status}</dd></div>}
          </dl>
          {selectedNode.document && <>
            <div className="inspector-section"><span>NOTE METADATA</span>
              <p>{selectedNode.document.domain} / {selectedNode.document.filename}</p>
              {selectedNode.chunk && <p>Page {selectedNode.chunk.page_number} / characters {selectedNode.chunk.start_char}-{selectedNode.chunk.end_char}</p>}
              <div className="inspector-tags">{selectedNode.document.tags.map((tag) => <i key={tag}>{tag}</i>)}</div>
              {!!selectedNode.document.sections.length && <div className="document-sections">
                <small>GEMINI SECTIONS</small>
                {selectedNode.document.sections.map((section) => <article key={`${section.title}-${section.page_start}`}>
                  <b>{section.title}</b><span>Pages {section.page_start}-{section.page_end}</span>
                  <p>{section.summary}</p>
                </article>)}
              </div>}
              {selectedNode.document.source_url && <a href={selectedNode.document.source_url} target="_blank" rel="noreferrer">Open source -&gt;</a>}
            </div>
            <div className="inspector-section"><span>PASSAGE PREVIEW</span>
              {relatedChunks.slice(0, 3).map((chunk) => <p className="inspector-passage" key={chunk.id}>{chunk.text}</p>)}
            </div>
          </>}
          {selectedNode.topic && <div className="inspector-section"><span>TOPIC KEYWORDS</span>
            <div className="inspector-tags">{selectedNode.topic.keywords.map((tag) => <i key={tag}>{tag}</i>)}</div>
          </div>}
          {selectedNode.kind === "topic" && <div className="inspector-section topic-description" aria-live="polite">
            <span>{activeTopicSummary ? "AI TOPIC DESCRIPTION" : "TOPIC DESCRIPTION"}</span>
            <p>{activeTopicSummary?.text || topicDescription}</p>
            {summaryLoading && <small>AI is reviewing this topic's source passages...</small>}
            {summaryError && <small className="topic-ai-error">AI description unavailable; showing the source text summary.</small>}
            {activeTopicSummary?.points.length ? <ul>{activeTopicSummary.points.map((point, index) =>
              <li key={`${index}:${point}`}>{point}</li>)}</ul> : null}
            {activeTopicSummary && <small>Based on {activeTopicSummary.sources} source{activeTopicSummary.sources === 1 ? "" : "s"} in this topic</small>}
          </div>}
          <div className="inspector-section bridges-section"><span>VERIFIED CONNECTIONS / {connectedBridges.length}</span>
            {connectedBridges.slice(0, 5).map((connection) => {
              const left = chunks.get(connection.source_chunk_id);
              const right = chunks.get(connection.target_chunk_id);
              const leftDoc = left ? documents.get(left.document_id) : undefined;
              const rightDoc = right ? documents.get(right.document_id) : undefined;
              return <article className="bridge-inspector-card" key={connection.id}>
                <strong>{connection.title || connection.shared_concept}</strong>
                <small>{leftDoc?.title || leftDoc?.filename} &lt;-&gt; {rightDoc?.title || rightDoc?.filename}</small>
                <p>{connection.explanation}</p>
                <span>SIMILARITY {connection.similarity.toFixed(2)} / WEIGHT {connection.weight.toFixed(2)}</span>
              </article>;
            })}
            {!connectedBridges.length && <p className="inspector-muted">No verified bridges touch this node yet.</p>}
          </div>
        </div>}
      </aside>

      {modalOpen && <div className="source-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setModalOpen(false); }}>
        <form className="source-modal" onSubmit={submitSource}>
          <div className="modal-top"><div><span>ADD TO YOUR NETWORK</span><h2>New source</h2></div>
            <button type="button" onClick={() => setModalOpen(false)} aria-label="Close">X</button></div>
          <div className="source-mode-tabs" role="tablist" aria-label="Source type">
            {(["file", "url", "note"] as const).map((mode) => <button key={mode} type="button"
              role="tab" aria-selected={sourceMode === mode} className={sourceMode === mode ? "active" : ""}
              onClick={() => { setSourceMode(mode); setSourceError(""); }}>{mode === "file" ? "Upload file" : mode === "url" ? "Web URL" : "Paste note"}</button>)}
          </div>
          {sourceMode === "file" && <label>PDF, CSV, TSV, or text file
            <input key={sourceMode} name="file" type="file" required accept=".pdf,.txt,.text,.md,.markdown,.csv,.tsv,.tab,application/pdf,text/plain,text/markdown,text/csv,text/tab-separated-values" />
          </label>}
          {sourceMode === "url" && <label>Public web page URL
            <input key={sourceMode} name="url" type="url" required placeholder="https://example.com/article" />
          </label>}
          {sourceMode === "note" && <>
            <label>Title<input name="title" required maxLength={512} placeholder="e.g. Adaptive systems in nature" /></label>
            <label>Domain<input name="domain" required maxLength={256} placeholder="e.g. Biology, Economics, Design" /></label>
            <label>Content<textarea name="content" required maxLength={200000} rows={6} placeholder="Add the note or passage you want connected..." /></label>
            <label>Tags <small>OPTIONAL</small><input name="tags" placeholder="Gemini will add suggested tags" /></label>
          </>}
          {sourceError && <p className="form-error" role="alert">{sourceError}</p>}
          <div className="modal-footer"><p>Stored in the local demo folder.</p>
            <button type="submit" disabled={saving}>{saving ? "READING + ANALYZING..." : "ADD SOURCE ->"}</button></div>
        </form>
      </div>}
    </main>
  );
}
