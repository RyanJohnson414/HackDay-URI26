import { useCallback, useEffect, useState } from "react";
import { fetchGraph } from "./api/client";
import DiscoveryFeed from "./components/DiscoveryFeed";
import UploadPanel from "./components/UploadPanel";
import { emptyGraph, type GraphResponse, type IngestionResponse } from "./types";
import "./styles.css";


export default function App() {
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem("hauri-theme") === "dark");
  const [graph, setGraph] = useState<GraphResponse>(emptyGraph);
  const [backendState, setBackendState] = useState<"connecting" | "live" | "demo">("connecting");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);

  const refreshGraph = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetchGraph();
      if (!response.documents.length && !response.topics.length && !response.chunks.length) {
        setGraph(response);
        setBackendState("live");
        setNotice("No sources are stored yet. Upload a file or add a note to build your graph.");
      } else {
        setGraph(response);
        setBackendState("live");
        setNotice(response.documents.length > 0 && response.documents.every((document) =>
          document.tags.includes("starter-example"))
          ? "Starter example data is loaded from SQLite. Add your own file, URL, or note to extend the graph."
          : "");
      }
    } catch (error) {
      setGraph(emptyGraph);
      setBackendState("demo");
      setNotice(error instanceof Error ? `Backend unavailable: ${error.message}.` : "Backend unavailable.");
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

  const sourceCount = graph.documents.length;
  const topicCount = graph.topics.length;
  const connectionCount = graph.connections.length;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Consilience home"><span className="brand-mark">C</span><span>Consilience<span className="brand-light"> / knowledge graph</span></span></a>
        <div className="topbar-right">
          <span className={`backend-status status-${backendState}`}><i />{backendState === "live" ? "Backend connected" : backendState === "connecting" ? "Connecting" : "Backend offline"}</span>
          <button className="theme-toggle" onClick={() => setDarkMode((mode) => !mode)}>{darkMode ? "Light mode" : "Dark mode"}</button>
          <a className="network-view-link" href="/network">Full-screen network ↗</a>
          <button className="icon-button" onClick={() => void refreshGraph()} disabled={loading} title="Refresh graph" aria-label="Refresh graph">↻</button>
        </div>
      </header>

      <section className="hero" id="top">
        <div>
          <div className="eyebrow"><span /> RESEARCH, CONNECTED</div>
          <h1>Ideas make more sense<br /><em>when they meet.</em></h1>
          <p>Bring your documents together. Consilience surfaces the concepts they share and the unexpected links between them.</p>
        </div>
        <div className="hero-stats">
          <div><strong>{sourceCount.toString().padStart(2, "0")}</strong><span>sources</span></div>
          <div><strong>{topicCount.toString().padStart(2, "0")}</strong><span>topic groups</span></div>
          <div><strong>{connectionCount.toString().padStart(2, "0")}</strong><span>connections</span></div>
        </div>
      </section>

      {notice && <div className={`notice${backendState === "demo" ? " notice-warn" : ""}`} role="status">{notice}</div>}

      <section className="workspace-grid">
        <div className="workspace-main">
          <div className="section-heading graph-heading">
            <div><span className="section-eyebrow">THE KNOWLEDGE MAP</span><h2>Topics & connections</h2></div>
          <span className="live-label"><i /> {backendState === "demo" ? "Backend offline" : "Live graph"}</span>
          </div>
          <iframe key={graph.documents.map((document) => document.id).join(":")}
            className="network-preview-frame" src="/network?embedded=1"
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
      <footer className="footer"><span>Consilience</span><span>Explore the space between ideas.</span></footer>
    </main>
  );
}
