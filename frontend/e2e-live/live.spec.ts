import { expect, test, type Page } from '@playwright/test';

/*
 * Phase 2 — the real UI on the real pipeline:
 * Login -> Chat -> Upload -> Processing -> Ready -> Ask -> Streaming -> Citation -> Follow-up -> History.
 * Uses a throwaway account; the test deletes its conversation and document at the end
 * (the account row itself is removed by the operator script that runs this suite).
 */

const PASSWORD = 'Live-ui-pass9';
export const EMAIL = process.env.LIVE_EMAIL ?? `live-ui-${Date.now()}@example.com`;

function pdf(pages: string[][]): Buffer {
  const objs: [number, string][] = [];
  const kids: string[] = [];
  const fontId = 3 + 2 * pages.length;
  pages.forEach((lines, i) => {
    const pageId = 3 + 2 * i;
    const contentId = 4 + 2 * i;
    kids.push(`${pageId} 0 R`);
    const stream = 'BT /F1 11 Tf 14 TL 50 760 Td ' + lines.map((l) => `(${l}) Tj T*`).join(' ') + ' ET';
    objs.push([pageId, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents ${contentId} 0 R /Resources << /Font << /F1 ${fontId} 0 R >> >> >>`]);
    objs.push([contentId, `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`]);
  });
  objs.unshift([2, `<< /Type /Pages /Kids [${kids.join(' ')}] /Count ${pages.length} >>`]);
  objs.unshift([1, '<< /Type /Catalog /Pages 2 0 R >>']);
  objs.push([fontId, '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>']);
  objs.sort((a, b) => a[0] - b[0]);
  let out = '%PDF-1.4\n';
  const offsets: Record<number, number> = {};
  for (const [id, body] of objs) {
    offsets[id] = out.length;
    out += `${id} 0 obj\n${body}\nendobj\n`;
  }
  const xref = out.length;
  out += `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n`;
  for (const [id] of objs) out += `${String(offsets[id]).padStart(10, '0')} 00000 n \n`;
  out += `trailer\n<< /Size ${objs.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`;
  return Buffer.from(out, 'latin1');
}

const SHEET = pdf([
  ['Aurora X2 Product Sheet - Warranty and Support',
    'The Aurora X2 wireless air-quality sensor is covered by a warranty of 24 months from delivery.',
    'The warranty covers manufacturing defects. Water damage beyond the IP65 rating is not covered.'],
  ['Aurora X2 Product Sheet - Environmental Limits',
    'The Aurora X2 operating temperature range is -20 to 60 degrees Celsius.',
    'Storage temperature range is -30 to 70 degrees Celsius.'],
]);

const normalise = (s: string) => s.replace(/[   ]/g, ' ').replace(/[‑−]/g, '-');

interface Frame { t: number; phase: string | null; len: number }

/** Records, inside the page, every render of the progress label and of the newest answer's text. */
async function startRecorder(page: Page) {
  await page.evaluate(() => {
    const w = window as unknown as { __frames: Frame[]; __obs?: MutationObserver };
    w.__frames = [];
    const t0 = performance.now();
    const snap = () => {
      const msgs = document.querySelectorAll('.message-wrapper .message-content');
      const last = msgs[msgs.length - 1];
      const frame = {
        t: Math.round(performance.now() - t0),
        phase: document.querySelector('.stream-status')?.textContent ?? null,
        len: last?.querySelector('.markdown')?.textContent?.length ?? 0,
      };
      const prev = w.__frames[w.__frames.length - 1];
      if (!prev || prev.phase !== frame.phase || prev.len !== frame.len) w.__frames.push(frame);
    };
    w.__obs?.disconnect();
    w.__obs = new MutationObserver(snap);
    w.__obs.observe(document.body, { subtree: true, childList: true, characterData: true });
  });
}

async function ask(page: Page, text: string) {
  const before = await page.locator('.message-wrapper').count();
  await startRecorder(page);
  await page.locator('.composer-textarea').fill(text);
  await page.locator('.composer-textarea').press('Enter');
  // The user message and the assistant placeholder appear as soon as message.created arrives.
  await expect(page.locator('.message-wrapper')).toHaveCount(before + 2);
  const answer = page.locator('.message-wrapper').last().locator('.message-content');
  // Finished = this turn's progress label was shown and is gone again.
  const getFrames = () => page.evaluate(() => (window as unknown as { __frames: Frame[] }).__frames);
  await expect.poll(async () => {
    const f = await getFrames();
    const started = f.findIndex((x) => x.phase);
    return started >= 0 && f[f.length - 1].phase === null && f.length - 1 > started;
  }, { timeout: 180_000, intervals: [250] }).toBe(true);
  const frames = await getFrames();
  const phases = [...new Set(frames.map((f) => f.phase).filter(Boolean))] as string[];
  const lengths = [...new Set(frames.map((f) => f.len).filter((n) => n > 0))];
  return { answer, phases, lengths, frames };
}

test('real UI: upload -> ready -> streamed grounded answer -> citations -> follow-up -> history', async ({ page }) => {
  // ---- Login (account created, then a real password login)
  await page.goto('/');
  await page.getByText('Sign up with email').click();
  await page.getByPlaceholder('Full name').fill('Live UI Tester');
  await page.getByPlaceholder('Email address').fill(EMAIL);
  await page.getByPlaceholder('Password').fill(PASSWORD);
  await page.getByRole('button', { name: 'Create account' }).click();
  await expect(page.getByText('How can I help you today?')).toBeVisible();
  await page.locator('.sidebar-footer').click();
  await page.getByText('Log out').click();
  await page.getByRole('button', { name: 'Log in', exact: true }).click();
  await page.getByPlaceholder('Email address').fill(EMAIL);
  await page.getByPlaceholder('Password').fill(PASSWORD);
  await page.getByRole('button', { name: 'Log in', exact: true }).click();
  await expect(page.getByText('How can I help you today?')).toBeVisible();

  // ---- Upload -> Processing -> Ready
  if (!(await page.locator('.context-panel').isVisible())) await page.getByTitle('Documents').click();
  await page.locator('.upload-area input[type=file]').setInputFiles({ name: 'aurora_x2_sheet.pdf', mimeType: 'application/pdf', buffer: SHEET });
  const row = page.locator('.doc-row', { hasText: 'aurora_x2_sheet.pdf' });
  await expect(row).toBeVisible();
  const seen = new Set<string>();
  for (let i = 0; i < 600 && !seen.has('Ready'); i++) {
    seen.add(((await row.locator('.status-chip').textContent().catch(() => '')) ?? '').trim());
    await page.waitForTimeout(100);
  }
  expect(seen.has('Ready'), `status chips seen: ${[...seen].join(' -> ')}`).toBeTruthy();
  console.log(`document status sequence: ${[...seen].filter(Boolean).join(' -> ')}`);
  await expect(row.locator('.status-ready')).toBeVisible();

  // ---- Attach and ask a grounded question: streamed, cited
  await row.locator('input[type=checkbox]').check().catch(() => {});
  const q1 = await ask(page, 'How long is the warranty on the Aurora X2?');
  console.log(`Q1 phases rendered: ${q1.phases.join(' -> ')}; text lengths rendered: ${q1.lengths.join(',')}`);
  console.log(`Q1 frames: ${JSON.stringify(q1.frames)}`);
  // The pipeline's progress is visible while it works (Agent Core events -> backend -> UI).
  expect(q1.phases).toContain('Searching your documents…');
  expect(q1.phases).toContain('Writing…');
  await expect(q1.answer.locator('.markdown')).toContainText('24');
  await expect(q1.answer.locator('.citations')).toContainText('aurora_x2_sheet.pdf p.1');
  const marker = q1.answer.locator('.cite-marker').first();
  await expect(marker).toBeVisible();
  await marker.click();
  await expect(q1.answer.locator('.source-preview')).toContainText('24 months');
  await expect(q1.answer.locator('.source-preview')).toContainText('page 1');

  // ---- Follow-up resolved against the conversation, cited to page 2
  const q2 = await ask(page, 'And what about its operating temperature?');
  await expect(q2.answer.locator('.markdown')).toContainText('60');
  expect(normalise((await q2.answer.locator('.markdown').textContent()) ?? '')).toContain('-20');
  await expect(q2.answer.locator('.citations')).toContainText('p.2');
  await q2.answer.locator('.source-chip', { hasText: 'p.2' }).first().click();
  await expect(q2.answer.locator('.source-preview')).toContainText('-20 to 60');

  // ---- Token streaming: a longer answer must render progressively, not all at once.
  const q3 = await ask(page, 'Ignoring the documents, explain in about 150 words how air-quality sensors measure CO2.');
  console.log(`Q2 renders: ${q2.lengths.length}; Q3 renders: ${q3.lengths.length} (final length ${q3.lengths.at(-1)})`);
  // How finely the text arrives depends on the upstream model (some free models send one burst),
  // so require that the UI rendered at least one answer progressively.
  expect(Math.max(q1.lengths.length, q2.lengths.length, q3.lengths.length), 'answer rendered incrementally while streaming').toBeGreaterThan(3);

  // ---- Conversation history survives a reload (re-fetched from the backend)
  await page.reload();
  await expect(page.locator('.history-item').first()).toContainText('How long is the warranty on the Aurora X2?');
  await page.locator('.history-item').first().click();
  await expect(page.locator('.message-wrapper')).toHaveCount(6);
  const answers = page.locator('.message-wrapper').filter({ has: page.locator('.message-avatar.assistant') });
  await expect(answers.nth(0).locator('.citations')).toContainText('p.1');
  await expect(answers.nth(1).locator('.citations')).toContainText('p.2');
  await answers.nth(1).locator('.cite-marker').first().click();
  await expect(answers.nth(1).locator('.source-preview')).toContainText('-20 to 60');

  // ---- Clean up through the UI's API session (conversation + document)
  const token = await page.evaluate(async () => {
    const r = await fetch('/api/v1/auth/refresh', { method: 'POST', credentials: 'include' });
    return (await r.json()).access_token as string;
  });
  const h = { Authorization: `Bearer ${token}` };
  const convs = await (await page.request.get('/api/v1/conversations', { headers: h })).json();
  for (const c of convs.items) expect((await page.request.delete(`/api/v1/conversations/${c.id}`, { headers: h })).status()).toBe(204);
  const docs = await (await page.request.get('/api/v1/documents', { headers: h })).json();
  for (const d of docs.items) expect((await page.request.delete(`/api/v1/documents/${d.id}`, { headers: h })).status()).toBe(204);
});
