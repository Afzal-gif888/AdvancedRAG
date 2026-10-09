import type { CSSProperties } from 'react';
import type { User } from '../types';

export const inputStyle: CSSProperties = {
  width: '100%',
  padding: '12px',
  borderRadius: '6px',
  border: '1px solid var(--border-color)',
  background: 'var(--input-bg)',
  color: 'var(--text-primary)',
  fontSize: '1rem',
};

export const primaryButtonStyle: CSSProperties = {
  padding: '8px 16px',
  background: 'var(--btn-bg)',
  color: 'var(--btn-text)',
  borderRadius: '6px',
  fontWeight: 500,
  cursor: 'pointer',
  border: 'none',
};

export const dangerButtonStyle: CSSProperties = { ...primaryButtonStyle, background: '#dc2626', color: 'white' };

// Must match the backend's ALLOWED_UPLOAD_EXTENSIONS (images need OCR in RAG Core first).
export const ACCEPTED_FILE_TYPES = '.pdf,.txt,.md,.csv,.docx';
export const ACCEPTED_FILE_LABEL = 'PDF, TXT, MD, CSV, DOCX';
/** Mirrors the backend's MAX_UPLOAD_BYTES (25 MB). Checked before uploading: a browser cannot read
 * the server's 413 reply when the server stops reading an oversized body early (it reports a
 * network error instead), so the size is validated client-side; the server still enforces it. */
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

export function displayName(user: User | null): string {
  return user?.full_name || user?.email?.split('@')[0] || 'You';
}

export function initial(user: User | null): string {
  return displayName(user).charAt(0).toUpperCase();
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export const ERROR_TEXT: Record<string, string> = {
  AGENT_SERVICE_UNAVAILABLE: 'The AI service is temporarily unavailable.',
  AGENT_SERVICE_ERROR: 'The AI service could not complete this request.',
  EMPTY_RESPONSE: 'The AI service returned an empty response.',
  GENERATION_FAILED: 'The response could not be generated.',
  CANCELLED: 'Response stopped.',
  RAG_SERVICE_UNAVAILABLE: 'Document search is temporarily unavailable.',
  LLM_UNAVAILABLE: 'The language model is temporarily unavailable.',
  LLM_TIMEOUT: 'The language model took too long to respond.',
  LLM_RATE_LIMITED: 'The language model is busy. Please try again shortly.',
  INTERRUPTED: 'The response was interrupted.',
};

/** Compact relative time for the sidebar: "now", "5m", "3h", "Yesterday", "Mon", "12 Mar", "12 Mar 2024". */
export function formatRelative(iso: string, now: Date = new Date()): string {
  const then = new Date(iso);
  const seconds = Math.round((now.getTime() - then.getTime()) / 1000);
  if (seconds < 60) return 'now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (then.getTime() >= startOfToday) return `${Math.floor(seconds / 3600)}h`;
  if (then.getTime() >= startOfToday - 86_400_000) return 'Yesterday';
  if (then.getTime() >= startOfToday - 6 * 86_400_000) return then.toLocaleDateString(undefined, { weekday: 'short' });
  const sameYear = then.getFullYear() === now.getFullYear();
  return then.toLocaleDateString(undefined, sameYear ? { day: 'numeric', month: 'short' } : { day: 'numeric', month: 'short', year: 'numeric' });
}
