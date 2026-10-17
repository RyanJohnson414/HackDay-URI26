import { useState, type FormEvent } from "react";
import { ingestUrl, uploadDocument } from "../api/client";
import type { IngestionResponse } from "../types";

export default function UploadPanel({ onComplete }: { onComplete: (result: IngestionResponse) => void }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submitFile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const input = event.currentTarget.elements.namedItem("document") as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const result = await uploadDocument(file);
      input.value = "";
      onComplete(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function submitUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!url.trim()) return;
    setBusy(true);
    setError("");
    try {
      const result = await ingestUrl(url.trim());
      setUrl("");
      onComplete(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not read that URL");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="ingest-card">
      <div className="ingest-heading"><span className="ingest-icon">＋</span><div>
        <strong>Add a source</strong><span>PDF, text file, or public web page</span>
      </div></div>
      <form className="file-form" onSubmit={submitFile}>
        <label className="file-picker">
          <input name="document" type="file" accept=".pdf,.txt,.text,application/pdf,text/plain" disabled={busy} />
          <span>Choose a file</span>
        </label>
        <button className="button button-dark" disabled={busy}>Upload</button>
      </form>
      <div className="form-divider"><span />or add a URL<span /></div>
      <form className="url-form" onSubmit={submitUrl}>
        <input type="url" value={url} onChange={(event) => setUrl(event.target.value)}
          placeholder="https://example.com/article" disabled={busy} aria-label="Web page URL" />
        <button className="button button-light" disabled={busy || !url.trim()}>{busy ? "Working…" : "Add URL"}</button>
      </form>
      {error && <p className="form-error">{error}</p>}
      {busy && <p className="form-hint">Reading the source, creating chunks, and updating the graph…</p>}
    </section>
  );
}
