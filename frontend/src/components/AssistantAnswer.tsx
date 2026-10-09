import { useCallback, useMemo, useRef, useState } from 'react';
import { FileText } from 'lucide-react';
import Markdown from './Markdown';
import type { Citation } from '../types';

/** Stable key for a citation: the [n] it answers to, or its position on older messages. */
function keyOf(c: Citation, i: number): string {
  return c.source_id ? `s${c.source_id}` : `i${i}`;
}

function describe(c: Citation): string {
  return `${c.filename ?? 'document'}${c.page ? `, page ${c.page}` : ''}`;
}

/**
 * An assistant answer with its sources. [n] markers in the text become numbered badges; the
 * sources below carry the same numbers and expand to show the cited passage (snippet) and page.
 * Citation data comes only from the backend (Agent Core maps markers to retrieved chunks), so a
 * badge can never point at a source that was not retrieved.
 */
export default function AssistantAnswer({ messageId, content, citations }: { messageId: string; content: string; citations: Citation[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const bySource = useMemo(() => {
    const m = new Map<number, Citation>();
    citations.forEach((c) => c.source_id && m.set(c.source_id, c));
    return m;
  }, [citations]);
  const sourceIds = useMemo(() => new Set(bySource.keys()), [bySource]);

  const onCite = useCallback((n: number) => {
    setOpen(`s${n}`);
    window.setTimeout(() => listRef.current?.querySelector(`[data-source="s${n}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }), 0);
  }, []);
  const citeLabel = useCallback((n: number) => {
    const c = bySource.get(n);
    return c ? `Source ${n}: ${describe(c)}` : `Source ${n}`;
  }, [bySource]);

  const active = citations.find((c, i) => keyOf(c, i) === open);

  return (
    <>
      {content && <Markdown text={content} sourceIds={sourceIds} onCite={onCite} citeLabel={citeLabel} />}
      {citations.length > 0 && (
        <div className="citations" ref={listRef} aria-label="Sources">
          <span className="citations-label">Sources:</span>
          {citations.map((c, i) => {
            const key = keyOf(c, i);
            return (
              <button
                type="button"
                key={key}
                data-source={key}
                className={`doc-chip source-chip${open === key ? ' active' : ''}`}
                aria-expanded={open === key}
                aria-controls={`source-${messageId}`}
                onClick={() => setOpen(open === key ? null : key)}
                title={describe(c)}
              >
                {c.source_id ? <span className="source-num">{c.source_id}</span> : <FileText size={12} />}
                {c.filename ?? 'document'}{c.page ? ` p.${c.page}` : ''}
              </button>
            );
          })}
        </div>
      )}
      {active && (
        <div className="source-preview" id={`source-${messageId}`} role="region" aria-label={`Source ${active.source_id ?? ''} ${describe(active)}`}>
          <div className="source-preview-head">
            {active.source_id && <span className="source-num">{active.source_id}</span>}
            <strong>{active.filename ?? 'document'}</strong>
            {active.page ? <span>page {active.page}</span> : null}
          </div>
          {active.snippet ? <blockquote>{active.snippet}{active.snippet.length >= 300 ? '…' : ''}</blockquote>
            : <p className="source-preview-empty">No excerpt available for this source.</p>}
        </div>
      )}
    </>
  );
}
