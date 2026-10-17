import type { GraphResponse } from "../types";

export default function DiscoveryFeed({ graph }: { graph: GraphResponse }) {
  const chunks = new Map(graph.chunks.map((chunk) => [chunk.id, chunk]));
  const documents = new Map(graph.documents.map((document) => [document.id, document.filename]));
  const connections = [...graph.connections].sort((a, b) => b.weight - a.weight).slice(0, 4);

  return (
    <section className="discovery-panel">
      <div className="section-heading">
        <div><span className="section-eyebrow">CROSS-DOMAIN DISCOVERIES</span><h2>Connections to explore</h2></div>
        <span className="count-badge">{graph.connections.length}</span>
      </div>
      {connections.length ? <div className="discovery-list">{connections.map((connection) => {
        const left = chunks.get(connection.source_chunk_id);
        const right = chunks.get(connection.target_chunk_id);
        return (
          <article className="discovery-item" key={connection.id}>
            <div className="discovery-score">{Math.round(connection.weight * 100)}<small>%</small></div>
            <div className="discovery-copy">
              <strong>{connection.title || connection.shared_concept}</strong>
              <p>{connection.explanation}</p>
              <div className="discovery-sources">
                <span>{left ? documents.get(left.document_id) : "Source"}</span>
                <b>↔</b>
                <span>{right ? documents.get(right.document_id) : "Source"}</span>
              </div>
            </div>
          </article>
        );
      })}</div> : <p className="muted discovery-empty">No cross-document connections yet. Add another source to start comparing ideas.</p>}
    </section>
  );
}
