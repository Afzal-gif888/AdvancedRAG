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

/** Statuses meaning "the backend is not answering right now" (cold start, restart, overload), not "you are signed out". */
const TRANSIENT_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);
const REFRESH_RETRY_DELAYS_MS = [1500, 3000, 6000];
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** Why the last refresh gave no session: the cookie is gone/invalid ('expired'), or the server did not answer ('unavailable'). */
export type RefreshOutcome = 'ok' | 'expired' | 'unavailable';

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
  if (TRANSIENT_STATUS.has(response.status)) {
    // The proxy answered, but the backend behind it did not (typically a free-tier service waking from sleep).
    return new ApiError(response.status, 'SERVICE_STARTING', 'The service is starting up after a quiet period. This can take up to a minute; please try again shortly.');
  }
  if (response.status === 404 || response.status === 405) {
    return new ApiError(response.status, 'ENDPOINT_NOT_FOUND', 'The server could not be reached at this address. If this persists, the deployment is misconfigured.');
  }
  return new ApiError(response.status, 'HTTP_ERROR', 'Something went wrong. Please try again.');
}

/**
 * The access token lives only in memory. The refresh token is an HttpOnly cookie the
 * browser sends to /auth/* automatically — JavaScript never sees it.
 */
/**
 * A non-secret marker, "this browser has signed in before". It only decides whether the app tries the
 * HttpOnly refresh cookie at start-up: a first-time visitor has no session, so the request would be a
 * guaranteed 401 that the browser prints as a red console error. The marker grants nothing; the cookie
 * and the access token still do all the authenticating.
 */
const SESSION_HINT = 'advancedrag.signed-in';

function setSessionHint(signedIn: boolean): void {
  try {
    if (signedIn) localStorage.setItem(SESSION_HINT, '1');
    else localStorage.removeItem(SESSION_HINT);
  } catch {
    // storage blocked (private mode): the app then simply always tries the cookie
  }
}

function hasSessionHint(): boolean {
  try {
    return localStorage.getItem(SESSION_HINT) !== null;
  } catch {
    return true;
  }
}

class ApiClient {
  private accessToken: string | null = null;
  private refreshInFlight: Promise<TokenResponse | null> | null = null;
  lastRefreshOutcome: RefreshOutcome = 'ok';
  onSessionExpired: (() => void) | null = null;

  setAccessToken(token: string | null) {
    this.accessToken = token;
  }

  /** False for a browser that has never signed in: start-up can skip the pointless refresh call. */
  mayHaveSession(): boolean {
    return hasSessionHint();
  }

  /**
   * Single-flight within a tab, and serialized across tabs with the Web Locks API, so tabs
   * never race each other's refresh-token rotation. (The server also tolerates a just-rotated
   * token for a short grace window as a fallback for browsers without Web Locks.)
   */
  refresh(): Promise<TokenResponse | null> {
    if (!this.refreshInFlight) {
      const run = async (): Promise<TokenResponse | null> => {
        // Only 401/403 mean "no valid session". A network error or a 5xx/gateway error means the server is not
        // answering (waking up, restarting): retry a few times with backoff and, if it is still down, report
        // 'unavailable' so the caller does NOT sign the user out.
        for (let attempt = 0; attempt <= REFRESH_RETRY_DELAYS_MS.length; attempt++) {
          try {
            const response = await fetch(`${API_BASE_URL}/auth/refresh`, { method: 'POST', credentials: 'include' });
            if (response.ok) {
              const data = (await response.json()) as TokenResponse;
              this.accessToken = data.access_token;
              this.lastRefreshOutcome = 'ok';
              setSessionHint(true);
              return data;
            }
            this.accessToken = null;
            if (response.status === 401 || response.status === 403) {
              this.lastRefreshOutcome = 'expired';
              setSessionHint(false); // definitely no session
              return null;
            }
            this.lastRefreshOutcome = 'unavailable';
            if (!TRANSIENT_STATUS.has(response.status)) return null; // 404/405...: retrying cannot help
          } catch {
            this.lastRefreshOutcome = 'unavailable'; // network error
          }
          if (attempt < REFRESH_RETRY_DELAYS_MS.length) await sleep(REFRESH_RETRY_DELAYS_MS[attempt]);
        }
        return null;
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
        if (this.lastRefreshOutcome === 'unavailable') {
          // The session may well be valid; the server just could not confirm it. Do not sign the user out.
          throw new ApiError(503, 'SERVICE_STARTING', 'The service is starting up after a quiet period. Please try again in a moment.');
        }
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
    setSessionHint(true);
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
    setSessionHint(true);
    return data;
  }

  async logout() {
    try {
      await this.send('/auth/logout', { method: 'POST' });
    } finally {
      this.accessToken = null;
      setSessionHint(false);
    }
  }

  async logoutAll() {
    await this.json<void>('/auth/logout-all', { method: 'POST' });
    this.accessToken = null;
    setSessionHint(false);
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
    if (res.status === 401 && this.lastRefreshOutcome !== 'unavailable') this.onSessionExpired?.();
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

  /** Upload a new version of a ready document; the current version keeps working until the new one is indexed. */
  async replaceDocument(id: string, file: File): Promise<DocumentInfo> {
    const form = new FormData();
    form.append('file', file);
    return this.json<DocumentInfo>(`/documents/${id}/replace`, { method: 'POST', body: form });
  }

  listDocumentVersions(id: string) {
    return this.json<DocumentInfo[]>(`/documents/${id}/versions`);
  }

  restoreDocumentVersion(id: string) {
    return this.json<DocumentInfo>(`/documents/${id}/restore`, { method: 'POST' });
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
