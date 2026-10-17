import type { GraphResponse, IngestionResponse, TopicSummary } from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP status as the useful fallback message.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function fetchGraph(): Promise<GraphResponse> {
  return request<GraphResponse>("/graph");
}

export function summarizeTopic(topicId: string): Promise<TopicSummary> {
  return request<TopicSummary>(`/graph/topics/${encodeURIComponent(topicId)}/summary`, { method: "POST" });
}

export function uploadDocument(file: File): Promise<IngestionResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<IngestionResponse>("/documents/upload", { method: "POST", body: form });
}

export function ingestUrl(url: string): Promise<IngestionResponse> {
  return request<IngestionResponse>("/documents/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
}

export interface NoteInput {
  title: string;
  domain: string;
  content: string;
  tags: string[];
}

export function createNote(note: NoteInput): Promise<IngestionResponse> {
  return request<IngestionResponse>("/documents/note", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(note),
  });
}
