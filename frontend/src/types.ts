export interface User {
  id: string;
  email: string;
  full_name: string | null;
  is_verified: boolean;
  plan: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
  snippet?: string | null;
}

export interface Attachment {
  id: string;
  filename: string;
}

export interface Citation {
  /** The n of the [n] marker in the answer text (absent on older messages). */
  source_id?: number | null;
  document_id: string | null;
  filename: string | null;
  page: number | null;
  chunk_id?: string | null;
  snippet: string | null;
}

export type MessageStatus = 'pending' | 'streaming' | 'completed' | 'failed' | 'cancelled';

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  status: MessageStatus;
  attachments: Attachment[];
  citations: Citation[];
  error_code: string | null;
  created_at: string;
}

export type DocumentStatus = 'uploaded' | 'processing' | 'ready' | 'failed' | 'superseded';

export interface DocumentInfo {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  status: DocumentStatus;
  processing_error: string | null;
  /** How the text was obtained and what could not be made searchable (null until indexing finished). */
  extraction?: ExtractionReport | null;
  version?: number;
  logical_id?: string | null;
  replaces_id?: string | null;
  /** True when the upload matched a document you already have (no new work was queued). */
  duplicate?: boolean;
  created_at: string;
  updated_at: string;
}

export interface ExtractionWarning {
  page: number | null;
  code: string;
  message: string;
}

export interface ExtractionReport {
  method: 'native' | 'ocr' | 'mixed';
  /** Some content is not searchable although the document is ready. */
  partial: boolean;
  total_pages: number | null;
  native_pages: number;
  ocr_pages: number;
  blank_pages: number[];
  unprocessed_pages: number[];
  failed_pages: number[];
  /** Tables turned into structured rows (only when PDF_TABLES is enabled): kind is lines | text | ocr_layout. */
  tables?: { page: number; kind: string; rows: number; columns: number; header: boolean; continued: boolean }[];
  warnings: ExtractionWarning[];
}

export interface ShareCreated {
  token: string;
  url: string;
  created_at: string;
  expires_at: string | null;
}

export interface ShareStatus {
  active: boolean;
  created_at: string | null;
  expires_at: string | null;
}

export interface SharedConversation {
  title: string;
  shared_at: string;
  messages: { role: 'user' | 'assistant'; content: string; created_at: string }[];
}

export type Theme = 'dark' | 'light' | 'system';

export interface UserSettings {
  theme: Theme;
}

export interface StreamEvent {
  event: string;
  data: Record<string, unknown>;
}
