import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import { CITE_PREFIX, linkCitations } from '../citations';
import remarkGfm from 'remark-gfm';
import { Check, Copy } from 'lucide-react';

/**
 * Safe Markdown for assistant output. Raw HTML in the source is never rendered (react-markdown
 * escapes it; no rehype-raw), and react-markdown's default URL transform drops `javascript:`
 * and other unsafe link protocols.
 */

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textOf).join('');
  if (node && typeof node === 'object' && 'props' in node) return textOf((node as { props: { children?: ReactNode } }).props.children);
  return '';
}

function CodeBlock({ language, children }: { language: string | null; children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const code = textOf(children).replace(/\n$/, '');
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: user can still select the text */
    }
  };
  return (
    <div className="code-block">
      <div className="code-block-header">
        <span>{language ?? 'code'}</span>
        <button onClick={copy} aria-label="Copy code">
          {copied ? <Check size={14} /> : <Copy size={14} />} {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  );
}

function buildComponents(onCite?: (n: number) => void, citeLabel?: (n: number) => string): Components {
  return {
    a: ({ href, children }) => {
      if (href?.startsWith(CITE_PREFIX)) {
        const n = Number(href.slice(CITE_PREFIX.length));
        return (
          <button type="button" className="cite-marker" onClick={() => onCite?.(n)} aria-label={citeLabel?.(n) ?? `Source ${n}`}>
            {n}
          </button>
        );
      }
      return <a href={href} target="_blank" rel="noopener noreferrer nofollow">{children}</a>;
    },
    // Images are never loaded: a model (or a prompt-injected document) could otherwise make the
    // browser request an attacker URL carrying conversation data.
    img: ({ alt, src }) => <span className="md-image-blocked" title={src ?? ''}>[image{alt ? `: ${alt}` : ''}]</span>,
    // Fenced blocks arrive as <pre><code class="language-x">; render them via CodeBlock.
    pre: ({ children }) => <>{children}</>,
    code: ({ className, children }) => {
      const match = /language-([\w+#-]+)/.exec(className ?? '');
      const isBlock = !!match || textOf(children).includes('\n');
      if (!isBlock) return <code className="inline-code">{children}</code>;
      return <CodeBlock language={match ? match[1] : null}>{children}</CodeBlock>;
    },
    table: ({ children }) => (
      <div className="table-wrap"><table>{children}</table></div>
    ),
  };
}

const plainComponents = buildComponents();

interface Props {
  text: string;
  /** Source numbers this answer actually cites; their [n] markers become clickable badges. */
  sourceIds?: ReadonlySet<number>;
  onCite?: (n: number) => void;
  citeLabel?: (n: number) => string;
}

export default function Markdown({ text, sourceIds, onCite, citeLabel }: Props) {
  const linked = sourceIds && sourceIds.size > 0;
  const components = useMemo(() => (linked ? buildComponents(onCite, citeLabel) : plainComponents), [linked, onCite, citeLabel]);
  const source = useMemo(() => (linked ? linkCitations(text, sourceIds) : text), [linked, text, sourceIds]);
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components} skipHtml>
        {source}
      </ReactMarkdown>
    </div>
  );
}
