import { useEffect } from 'react';
import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import type { DocumentStatus } from '../types';

const STATUS_LABEL: Record<DocumentStatus | 'uploading', string> = {
  uploading: 'Uploading',
  uploaded: 'Queued',
  processing: 'Processing',
  ready: 'Ready',
  failed: 'Failed',
  superseded: 'Older version',
};

export function StatusChip({ status, progress }: { status: DocumentStatus | 'uploading'; progress?: number }) {
  const label = status === 'uploading' && progress !== undefined ? `Uploading ${Math.round(progress * 100)}%` : STATUS_LABEL[status];
  return <span className={`status-chip status-${status}`}>{label}</span>;
}

export function FormMessage({ error, success }: { error?: string | null; success?: string | null }) {
  if (error) return <div className="form-error" role="alert">{error}</div>;
  if (success) return <div className="form-success">{success}</div>;
  return null;
}

/** In-app confirmation used everywhere instead of the browser's confirm(): title, message,
 *  Cancel + a red action. Escape / Cancel / clicking outside dismiss it (not while busy). */
export function ConfirmDialog({ title, children, confirmLabel, busyLabel = 'Deleting…', busy, error, onConfirm, onCancel }: {
  title: string; children: ReactNode; confirmLabel: string; busyLabel?: string; busy?: boolean; error?: string | null;
  onConfirm: () => void; onCancel: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) onCancel(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, onCancel]);
  return (
    <div className="modal-overlay" onClick={() => !busy && onCancel()}>
      <div className="confirm-dialog" onClick={(e) => e.stopPropagation()} role="alertdialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        <div className="confirm-body">{children}</div>
        {error && <div className="form-error" role="alert">{error}</div>}
        <div className="confirm-actions">
          <button className="confirm-cancel" onClick={onCancel} disabled={busy} autoFocus>Cancel</button>
          <button className="confirm-danger" onClick={onConfirm} disabled={busy}>{busy ? busyLabel : confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}

export function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={title}>
        <div className="modal-header">
          <h2>{title}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><X size={20} /></button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}

