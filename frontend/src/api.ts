import type {
  Conversation,
  DocumentInfo,
  Message,
  Page,
  ShareCreated,
  SharedConversation,
  ShareStatus,
  StreamEvent,
  TokenResponse,
  User,
  UserSettings,
} from './types';

// Same-origin by default (Vite dev server proxies /api to the backend).
// `||` (not `??`): an empty VITE_API_BASE_URL in .env must fall back to the same-origin default.
const API_BASE_URL = ((import.meta.env.VITE_API_BASE_URL as string | undefined) || '/api/v1').replace(/\/$/, '');

export class ApiError extends Error {
  status: number;
  code: string;
  details?: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

type Json = Record<string, unknown> | unknown[];

interface RequestOptions {
  method?: string;
  body?: Json | FormData;
  auth?: boolean;
  signal?: AbortSignal;
}

async function toApiError(response: Response): Promise<ApiError> {
  const body = await response.json().catch(() => null);
  const err = body?.error;
  if (err) return new ApiError(response.status, err.code, err.message, err.details);
  return new ApiError(response.status, 'HTTP_ERROR', 'Something went wrong. Please try again.');
}

/**
 * The access token lives only in memory. The refresh token is an HttpOnly cookie the
 * browser sends to /auth/* automatically — JavaScript never sees it.
 */
class ApiClient {
  private accessToken: string | null = null;
  private refreshInFlight: Promise<TokenResponse | null> | null = null;
  onSessionExpired: (() => void) | null = null;

  setAccessToken(token: string | null) {
    this.accessToken = token;
  }

  /**
   * Single-flight within a tab, and serialized across tabs with the Web Locks API, so tabs
   * never race each other's refresh-token rotation. (The server also tolerates a just-rotated
   * token for a short grace window as a fallback for browsers without Web Locks.)
   */
  refresh(): Promise<TokenResponse | null> {
    if (!this.refreshInFlight) {
      const run = async () => {
        try {
          const response = await fetch(`${API_BASE_URL}/auth/refresh`, { method: 'POST', credentials: 'include' });
          if (!response.ok) {
            this.accessToken = null;
            return null;
          }
          const data = (await response.json()) as TokenResponse;
          this.accessToken = data.access_token;
          return data;
        } catch {
          return null;
        }
      };
      const locks = (navigator as Navigator & { locks?: LockManager }).locks;
      const task = locks ? (locks.request('advancedrag-auth-refresh', run) as Promise<TokenResponse | null>) : run();
      this.refreshInFlight = task.finally(() => {
        this.refreshInFlight = null;
      });
    }
    return this.refreshInFlight;
  }

  private async send(path: string, opts: RequestOptions): Promise<Response> {
    const headers: Record<string, string> = {};
    let body: BodyInit | undefined;
    if (opts.body instanceof FormData) {
      body = opts.body; // browser sets the multipart boundary
    } else if (opts.body !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(opts.body);
    }
    if (opts.auth !== false && this.accessToken) headers['Authorization'] = `Bearer ${this.accessToken}`;
    return fetch(`${API_BASE_URL}${path}`, {
      method: opts.method ?? 'GET',
      headers,
      body,
      credentials: 'include',
      signal: opts.signal,
    });
  }

  /** Sends a request; on 401 refreshes once and retries. */
  private async raw(path: string, opts: RequestOptions = {}): Promise<Response> {
    let response = await this.send(path, opts);
    if (response.status === 401 && opts.auth !== false) {
      const refreshed = await this.refresh();
      if (!refreshed) {
        this.onSessionExpired?.();
        throw await toApiError(response);
      }
      response = await this.send(path, opts);
    }
    if (!response.ok) throw await toApiError(response);
    return response;
  }

  private async json<T>(path: string, opts: RequestOptions = {}): Promise<T> {
    const response = await this.raw(path, opts);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  // ------------------------------------------------------------------ auth
  /** Email + password. Starts the session (access token here, refresh token as HttpOnly cookie). */
  async login(email: string, password: string): Promise<TokenResponse> {
    const data = await this.json<TokenResponse>('/auth/login', { method: 'POST', body: { email, password }, auth: false });
    this.accessToken = data.access_token;
    return data;
  }

  /** Creates the account (name, email, password) and starts the session. */
  async register(email: string, password: string, fullName: string): Promise<TokenResponse> {
    const data = await this.json<TokenResponse>('/auth/register', {
      method: 'POST',
      body: { email, password, full_name: fullName },
      auth: false,
    });
    this.accessToken = data.access_token;
    return data;
  }

  async logout() {
    try {
      await this.send('/auth/logout', { method: 'POST' });
    } finally {
      this.accessToken = null;
    }
  }

  async logoutAll() {
    await this.json<void>('/auth/logout-all', { method: 'POST' });
    this.accessToken = null;
  }

  me() {
    return this.json<User>('/auth/me');
  }

  forgotPassword(email: string) {
    return this.json<{ message: string }>('/auth/forgot-password', { method: 'POST', body: { email }, auth: false });
  }

  resetPassword(token: string, newPassword: string) {
    return this.json<{ message: string }>('/auth/reset-password', {
      method: 'POST',
      body: { token, new_password: newPassword },
      auth: false,
    });
  }

  // ------------------------------------------------------------------ profile & settings
  updateProfile(fullName: string) {
    return this.json<User>('/profile', { method: 'PATCH', body: { full_name: fullName } });
  }

  clearAllData() {
    return this.json<void>('/profile/data', { method: 'DELETE' });
  }

  getSettings() {
    return this.json<UserSettings>('/settings');
  }

  updateSettings(patch: Partial<UserSettings>) {
    return this.json<UserSettings>('/settings', { method: 'PATCH', body: patch });
  }

  // ------------------------------------------------------------------ conversations
  listConversations(limit = 30, offset = 0) {
    return this.json<Page<Conversation>>(`/conversations?limit=${limit}&offset=${offset}`);
  }

  createConversation(title?: string) {
    return this.json<Conversation>('/conversations', { method: 'POST', body: title ? { title } : {} });
  }

  renameConversation(id: string, title: string) {
    return this.json<Conversation>(`/conversations/${id}`, { method: 'PATCH', body: { title } });
  }

  deleteConversation(id: string) {
    return this.json<void>(`/conversations/${id}`, { method: 'DELETE' });
  }

  searchChats(q: string, signal?: AbortSignal) {
    return this.json<Page<Conversation>>(`/search/chats?q=${encodeURIComponent(q)}&limit=20`, { signal });
  }

  // ------------------------------------------------------------------ messages
  /** Newest-first page; `offset` counts back from the latest message. */
  listMessages(conversationId: string, limit = 50, offset = 0) {
    return this.json<Page<Message>>(`/conversations/${conversationId}/messages?limit=${limit}&offset=${offset}&order=desc`);
  }

  /** POSTs and yields Server-Sent Events until the stream ends. */
  async *streamMessage(conversationId: string, content: string, documentIds: string[], signal?: AbortSignal) {
    const response = await this.raw(`/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      body: { content, document_ids: documentIds },
      signal,
    });
    yield* readEventStream(response);
  }

  async *retryMessage(conversationId: string, messageId: string, signal?: AbortSignal) {
    const response = await this.raw(`/conversations/${conversationId}/messages/${messageId}/retry`, {
      method: 'POST',
      signal,
    });
    yield* readEventStream(response);
  }

  // ------------------------------------------------------------------ documents
  listDocuments(limit = 50, offset = 0) {
    return this.json<Page<DocumentInfo>>(`/documents?limit=${limit}&offset=${offset}`);
  }

  getDocument(id: string) {
    return this.json<DocumentInfo>(`/documents/${id}`);
  }

  /** XHR (not fetch) so upload progress can be reported. Retries once after a token refresh. */
  async uploadDocument(file: File, onProgress?: (fraction: number) => void): Promise<DocumentInfo> {
    const attempt = () =>
      new Promise<{ status: number; body: unknown }>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API_BASE_URL}/documents`);
        xhr.withCredentials = true;
        if (this.accessToken) xhr.setRequestHeader('Authorization', `Bearer ${this.accessToken}`);
        xhr.responseType = 'json';
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
        };
        xhr.onload = () => resolve({ status: xhr.status, body: xhr.response });
        xhr.onerror = () => reject(new ApiError(0, 'NETWORK_ERROR', 'Upload failed. Please check your connection.'));
        const form = new FormData();
        form.append('file', file);
        xhr.send(form);
      });
    let res = await attempt();
    if (res.status === 401 && (await this.refresh())) res = await attempt();
    if (res.status === 401) this.onSessionExpired?.();
    if (res.status < 200 || res.status >= 300) {
      const err = (res.body as { error?: { code: string; message: string; details?: unknown } } | null)?.error;
      throw err ? new ApiError(res.status, err.code, err.message, err.details) : new ApiError(res.status, 'HTTP_ERROR', 'Upload failed.');
    }
    return res.body as DocumentInfo;
  }

  deleteDocument(id: string) {
    return this.json<void>(`/documents/${id}`, { method: 'DELETE' });
  }

  reprocessDocument(id: string) {
    return this.json<DocumentInfo>(`/documents/${id}/reprocess`, { method: 'POST' });
  }

  listConversationDocuments(conversationId: string) {
    return this.json<DocumentInfo[]>(`/conversations/${conversationId}/documents`);
  }

  attachDocuments(conversationId: string, documentIds: string[]) {
    return this.json<DocumentInfo[]>(`/conversations/${conversationId}/documents`, {
      method: 'POST',
      body: { document_ids: documentIds },
    });
  }

  detachDocument(conversationId: string, documentId: string) {
    return this.json<void>(`/conversations/${conversationId}/documents/${documentId}`, { method: 'DELETE' });
  }

  // ------------------------------------------------------------------ sharing
  getShareStatus(conversationId: string) {
    return this.json<ShareStatus>(`/conversations/${conversationId}/share`);
  }

  createShare(conversationId: string) {
    return this.json<ShareCreated>(`/conversations/${conversationId}/share`, { method: 'POST', body: {} });
  }

  revokeShare(conversationId: string) {
    return this.json<void>(`/conversations/${conversationId}/share`, { method: 'DELETE' });
  }

  getSharedConversation(token: string) {
    return this.json<SharedConversation>(`/shares/${encodeURIComponent(token)}`, { auth: false });
  }
}

async function* readEventStream(response: Response): AsyncGenerator<StreamEvent> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = 'message';
      const dataLines: string[] = [];
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length) {
        try {
          yield { event, data: JSON.parse(dataLines.join('\n')) };
        } catch {
          // ignore malformed frames
        }
      }
    }
  }
}

export const api = new ApiClient();

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.code === 'VALIDATION_ERROR' && Array.isArray(err.details) && err.details.length) {
      const first = err.details[0] as { message?: string };
      return (first.message ?? err.message).replace(/^Value error, /, '');
    }
    return err.message;
  }
  return 'Network error. Please check your connection.';
}
