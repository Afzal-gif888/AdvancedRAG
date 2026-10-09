/** Citation marker handling for assistant answers (pure functions, unit-testable). */

// Source markers as models write them: [2], [1, 3], [1][4] and full-width 【2】 (never a link's text).
const MARKER = /[[【]\s*(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\s*[\]】](?!\()/g;
// Fenced code blocks and inline code are left untouched.
const CODE = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g;
export const CITE_PREFIX = '#cite-';

/**
 * Turn [n] markers that refer to a real source of this answer into internal `#cite-n` links,
 * rendered as numbered badges. Unknown numbers stay plain text (they were never a source).
 */
export function linkCitations(text: string, sourceIds: ReadonlySet<number>): string {
  if (!sourceIds.size) return text;
  return text
    .split(CODE)
    .map((part, i) => (i % 2 === 1 ? part : part.replace(MARKER, (whole, nums: string) => {
      const ns = nums.split(/[,;]/).map((n) => Number(n.trim()));
      if (!ns.some((n) => sourceIds.has(n))) return whole;
      return ns.map((n) => (sourceIds.has(n) ? `[${n}](${CITE_PREFIX}${n})` : `\\[${n}\\]`)).join('');
    })))
    .join('');
}
