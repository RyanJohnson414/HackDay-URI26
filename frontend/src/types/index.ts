export interface DocumentRecord {
  id: string;
  filename: string;
  title: string;
  domain: string;
  tags: string[];
  sections: DocumentSection[];
  media_type: string;
  source_url: string | null;
  status: string;
  error: string | null;
  created_at: string;
  chunk_count: number;
}

export interface DocumentSection {
  title: string;
  summary: string;
  page_start: number;
  page_end: number;
}

export interface GraphChunk {
  id: string;
  document_id: string;
  topic_id: string | null;
  page_number: number;
  text: string;
  start_char: number;
  end_char: number;
  source_url: string | null;
  centroid_distance: number | null;
}

export interface GraphConnection {
  id: string;
  source_chunk_id: string;
  target_chunk_id: string;
  similarity: number;
  strength: number;
  weight: number;
  title: string;
  explanation: string;
  shared_concept: string;
}

export interface GraphTopic {
  id: string;
  label: string;
  keywords: string[];
  chunk_ids: string[];
}

export interface TopicLink {
  source_topic_id: string;
  target_topic_id: string;
  similarity: number;
}

export interface TopicSummary {
  topic_id: string;
  subject: string;
  summary: string;
  key_points: string[];
  source_count: number;
}

export interface GraphResponse {
  documents: DocumentRecord[];
  chunks: GraphChunk[];
  connections: GraphConnection[];
  topics: GraphTopic[];
  related_topics: TopicLink[];
}

export const emptyGraph: GraphResponse = {
  documents: [],
  chunks: [],
  connections: [],
  topics: [],
  related_topics: [],
};

export interface IngestionResponse {
  document: DocumentRecord;
  connection_count: number;
  topic_count: number;
  related_topic_count: number;
  warnings: string[];
}
