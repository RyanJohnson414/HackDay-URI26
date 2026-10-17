import { useCallback, useEffect, useState } from "react";
import { fetchGraph } from "./api/client";
import DiscoveryFeed from "./components/DiscoveryFeed";
import UploadPanel from "./components/UploadPanel";
import type { GraphResponse, IngestionResponse } from "./types";
import "./styles.css";

export const sampleGraph: GraphResponse = {
  documents: [
    { id: "doc-bio", filename: "Collective behavior.pdf", title: "Collective behavior", domain: "Biology", tags: ["biology"], media_type: "application/pdf", source_url: null, status: "complete", error: null, created_at: "2026-01-01", chunk_count: 2 },
    { id: "doc-network", filename: "Mesh networks.txt", title: "Mesh networks", domain: "Computer networks", tags: ["routing"], media_type: "text/plain", source_url: null, status: "complete", error: null, created_at: "2026-01-02", chunk_count: 2 },
    { id: "doc-city", filename: "Urban mobility.pdf", title: "Urban mobility", domain: "Urban planning", tags: ["transport"], media_type: "application/pdf", source_url: null, status: "complete", error: null, created_at: "2026-01-03", chunk_count: 1 },
    { id: "doc-space", filename: "Planetary systems.txt", title: "Planetary systems", domain: "Astronomy", tags: ["space"], media_type: "text/plain", source_url: null, status: "complete", error: null, created_at: "2026-01-04", chunk_count: 1 },
  ],
  chunks: [
    { id: "chunk-bio-1", document_id: "doc-bio", topic_id: "topic-bio", page_number: 2, text: "Each ant responds to nearby signals rather than a central commander. Simple local rules let the colony coordinate its work.", start_char: 180, end_char: 299, source_url: null, centroid_distance: 0.08 },
    { id: "chunk-bio-2", document_id: "doc-bio", topic_id: "topic-bio", page_number: 3, text: "Ants leave and follow local chemical trail signals, allowing the group to adapt without one ant directing every worker.", start_char: 420, end_char: 535, source_url: null, centroid_distance: 0.12 },
    { id: "chunk-net-1", document_id: "doc-network", topic_id: "topic-network", page_number: 1, text: "Each router reacts to local congestion instead of a central controller. Local routing lets the network adapt to changing traffic.", start_char: 0, end_char: 124, source_url: null, centroid_distance: 0.09 },
    { id: "chunk-net-2", document_id: "doc-network", topic_id: "topic-network", page_number: 2, text: "A node forwards packets using current local link conditions, so the network can keep operating when routes change.", start_char: 320, end_char: 430, source_url: null, centroid_distance: 0.16 },
    { id: "chunk-city-1", document_id: "doc-city", topic_id: "topic-city", page_number: 4, text: "Traffic signals adapt to local vehicle queues, easing congestion without requiring one central schedule for every street.", start_char: 750, end_char: 866, source_url: null, centroid_distance: 0.02 },
    { id: "chunk-space-1", document_id: "doc-space", topic_id: "topic-space", page_number: 1, text: "Repeated telescope observations reveal the positions and orbital periods of planets around a star.", start_char: 0, end_char: 95, source_url: null, centroid_distance: 0.01 },
  ],
  topics: [
    { id: "topic-bio", label: "Collective behavior", keywords: ["collective", "signals", "colony"], chunk_ids: ["chunk-bio-1", "chunk-bio-2"] },
    { id: "topic-network", label: "Distributed networks", keywords: ["routing", "network", "local"], chunk_ids: ["chunk-net-1", "chunk-net-2"] },
    { id: "topic-city", label: "Urban traffic", keywords: ["traffic", "queues", "congestion"], chunk_ids: ["chunk-city-1"] },
    { id: "topic-space", label: "Planetary systems", keywords: ["planets", "orbits", "telescope"], chunk_ids: ["chunk-space-1"] },
  ],
  connections: [
    { id: "edge-bio-net", source_chunk_id: "chunk-bio-1", target_chunk_id: "chunk-net-1", similarity: 0.64, strength: 0.93, weight: 0.81, title: "Coordination without central control", explanation: "Both systems adapt through local responses: ants respond to nearby signals, while routers react to local congestion.", shared_concept: "Decentralized local response" },
    { id: "edge-net-city", source_chunk_id: "chunk-net-2", target_chunk_id: "chunk-city-1", similarity: 0.59, strength: 0.82, weight: 0.73, title: "Local conditions guide flow", explanation: "Both networks adjust movement using conditions observed nearby.", shared_concept: "Local flow control" },
  ],
  related_topics: [
    { source_topic_id: "topic-network", target_topic_id: "topic-city", similarity: 0.71 },
  ],
};

export default function App() {
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem("hauri-theme") === "dark");
  const [graph, setGraph] = useState<GraphResponse>(sampleGraph);
  const [isDemo, setIsDemo] = useState(true);
  const [backendState, setBackendState] = useState<"connecting" | "live" | "demo">("connecting");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);

  const refreshGraph = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetchGraph();
      if (!response.documents.length && !response.topics.length && !response.chunks.length) {
        setGraph(sampleGraph);
        setIsDemo(true);
        setBackendState("demo");
        setNotice("Backend is connected. Showing a sample graph until you add your first source.");
      } else {
        setGraph(response);
        setIsDemo(false);
        setBackendState("live");
        setNotice("");
      }
    } catch (error) {
      setGraph(sampleGraph);
      setIsDemo(true);
      setBackendState("demo");
      setNotice(error instanceof Error ? `Backend unavailable: ${error.message}. Showing sample data.` : "Backend unavailable. Showing sample data.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refreshGraph(); }, [refreshGraph]);

  useEffect(() => {
    document.body.classList.toggle("dark-mode", darkMode);
    localStorage.setItem("hauri-theme", darkMode ? "dark" : "light");
    const syncTheme = (event: StorageEvent) => {
      if (event.key === "hauri-theme") setDarkMode(event.newValue === "dark");
    };
    window.addEventListener("storage", syncTheme);
    return () => window.removeEventListener("storage", syncTheme);
  }, [darkMode]);

  function handleIngest(result: IngestionResponse) {
    const details = result.warnings.length ? ` ${result.warnings.join(" ")}` :
      ` Found ${result.connection_count} connections across ${result.topic_count} topic groups.`;
    const message = `${result.document.filename} added.${details}`;
    void refreshGraph().then(() => setNotice(message));
  }

  const sourceCount = isDemo ? sampleGraph.documents.length : graph.documents.length;
  const topicCount = isDemo ? sampleGraph.topics.length : graph.topics.length;
  const connectionCount = isDemo ? sampleGraph.connections.length : graph.connections.length;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="HaURI home"><span className="brand-mark">H</span><span>HaURI<span className="brand-light"> / knowledge graph</span></span></a>
        <div className="topbar-right">
          <span className={`backend-status status-${backendState}`}><i />{backendState === "live" ? "Backend connected" : backendState === "connecting" ? "Connecting" : "Preview mode"}</span>
          <button className="theme-toggle" onClick={() => setDarkMode((mode) => !mode)}>{darkMode ? "Light mode" : "Dark mode"}</button>
          <a className="network-view-link" href="/network">Full-screen network ↗</a>
          <button className="icon-button" onClick={() => void refreshGraph()} disabled={loading} title="Refresh graph" aria-label="Refresh graph">↻</button>
        </div>
      </header>

      <section className="hero" id="top">
        <div>
          <div className="eyebrow"><span /> RESEARCH, CONNECTED</div>
          <h1>Ideas make more sense<br /><em>when they meet.</em></h1>
          <p>Bring your documents together. HaURI surfaces the concepts they share and the unexpected links between them.</p>
        </div>
        <div className="hero-stats">
          <div><strong>{sourceCount.toString().padStart(2, "0")}</strong><span>sources</span></div>
          <div><strong>{topicCount.toString().padStart(2, "0")}</strong><span>topic groups</span></div>
          <div><strong>{connectionCount.toString().padStart(2, "0")}</strong><span>connections</span></div>
        </div>
      </section>

      {notice && <div className={`notice${backendState === "demo" && !notice.startsWith("Backend is connected") ? " notice-warn" : ""}`} role="status">{notice}</div>}

      <section className="workspace-grid">
        <div className="workspace-main">
          <div className="section-heading graph-heading">
            <div><span className="section-eyebrow">THE KNOWLEDGE MAP</span><h2>Topics & connections</h2></div>
            <span className="live-label"><i /> {isDemo ? "Illustrative view" : "Live graph"}</span>
          </div>
          <iframe className="network-preview-frame" src="/network?embedded=1"
            title="Interactive knowledge network preview" />
          <DiscoveryFeed graph={graph} />
        </div>
        <aside className="workspace-side">
          <UploadPanel onComplete={handleIngest} />
          <section className="how-card">
            <span className="section-eyebrow">HOW TO READ THE MAP</span>
            <div className="how-row"><span className="how-number">01</span><p><b>Topic groups</b><br />Passages with similar meaning are collected together.</p></div>
            <div className="how-row"><span className="how-number">02</span><p><b>Discovered connections</b><br />A strong link means two sources share an idea.</p></div>
            <div className="how-row"><span className="how-number">03</span><p><b>Similar topics</b><br />Dashed links show neighboring themes worth exploring.</p></div>
          </section>
          <div className="privacy-note"><span>↳</span> Your sources are processed by the local backend and stored in its SQLite database.</div>
        </aside>
      </section>
      <footer className="footer"><span>HaURI26</span><span>Explore the space between ideas.</span></footer>
    </main>
  );
}
