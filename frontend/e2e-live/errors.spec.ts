import { expect, test, type Page, type Response } from '@playwright/test';

/*
 * Phase 3 — deliberate failures, seen from the browser. Each test asserts the user gets a clean,
 * specific message and that no API response or page text exposes internal details.
 *
 * Service-failure tests ("fault:") run against a fault-injection stack prepared by the operator
 * (LIVE_BASE_URL); the others run against the normal dev stack.
 */

const PASSWORD = 'Phase3-pass9';
const FORBIDDEN = ['127.0.0.1', 'localhost:', 'Traceback', 'httpx', 'ConnectError', 'ReadTimeout', 'freellmapi', 'FreeLLMAPI',
  'rag_unreachable', 'rag_timeout', 'llm_service', 'LLM_AUTH', 'pinecone', 'Pinecone', 'INTERNAL_SERVICE', 'sqlalchemy', 'asyncpg'];

/** Records every API response body seen by the page (stream bodies included). */
function watchResponses(page: Page) {
  const bodies: string[] = [];
  page.on('response', async (r: Response) => {
    if (!r.url().includes('/api/')) return;
    try { bodies.push(`${r.status()} ${r.url()} ${await r.text()}`); } catch { /* body unavailable (aborted) */ }
  });
  return bodies;
}

async function assertNoLeaks(page: Page, bodies: string[]) {
  await page.waitForTimeout(500);
  const visible = (await page.locator('body').innerText()) ?? '';
  for (const word of FORBIDDEN) {
    expect(visible, `page text must not contain "${word}"`).not.toContain(word);
    for (const b of bodies) expect(b.replace(/https?:\/\/(localhost|127\.0\.0\.1):\d+\/api\/[^\s]*/, '<url>'), `API body must not contain "${word}"`).not.toContain(word);
  }
}

async function signUp(page: Page) {
  const email = `phase3-ui-${Date.now()}-${Math.floor(Math.random() * 1e6)}@example.com`;
  await page.goto('/');
  await page.getByText('Sign up with email').click();
  await page.getByPlaceholder('Full name').fill('Phase Three');
  await page.getByPlaceholder('Email address').fill(email);
  await page.getByPlaceholder('Password').fill(PASSWORD);
  await page.getByRole('button', { name: 'Create account' }).click();
  await expect(page.getByText('How can I help you today?')).toBeVisible();
  return email;
}

async function openDocs(page: Page) {
  if (!(await page.locator('.context-panel').isVisible())) await page.getByTitle('Documents').click();
}

async function upload(page: Page, name: string, mimeType: string, buffer: Buffer) {
  await openDocs(page);
  await page.locator('.upload-area input[type=file]').setInputFiles({ name, mimeType, buffer });
}

async function send(page: Page, text: string) {
  await page.locator('.composer-textarea').fill(text);
  await page.locator('.composer-textarea').press('Enter');
}

async function expectFailedTurn(page: Page, message: string) {
  const last = page.locator('.message-wrapper').last();
  await expect(last.locator('.message-error')).toContainText(message, { timeout: 120_000 });
  await expect(last.getByRole('button', { name: 'Retry' })).toBeVisible();
  await expect(page.locator('.stream-status')).toHaveCount(0);
}

// ------------------------------------------------------------------ fault-injection stack

test('fault: FreeLLMAPI unavailable', async ({ page }) => {
  const bodies = watchResponses(page);
  await signUp(page);
  await send(page, 'What is a heat pump?');
  await expectFailedTurn(page, 'The language model is temporarily unavailable.');
  await assertNoLeaks(page, bodies);
});

test('fault: LLM timeout', async ({ page }) => {
  const bodies = watchResponses(page);
  await signUp(page);
  await send(page, 'What is a heat pump?');
  await expectFailedTurn(page, 'The language model took too long to respond.');
  await assertNoLeaks(page, bodies);
});

test('fault: RAG unavailable', async ({ page }) => {
  const bodies = watchResponses(page);
  await signUp(page);
  await upload(page, 'policy.txt', 'text/plain', Buffer.from('Refunds are processed within five business days.'));
  await expect(page.locator('.doc-row', { hasText: 'policy.txt' }).locator('.status-ready')).toBeVisible({ timeout: 60_000 });
  await send(page, 'How long do refunds take?');
  await expectFailedTurn(page, 'Document search is temporarily unavailable.');
  // Nothing was generated without the documents (no fabricated answer).
  await expect(page.locator('.message-wrapper').last().locator('.markdown')).toHaveCount(0);
  await assertNoLeaks(page, bodies);
});

// ------------------------------------------------------------------ normal dev stack

test('ui: invalid documents', async ({ page }) => {
  const bodies = watchResponses(page);
  await signUp(page);
  await upload(page, 'tool.exe', 'application/octet-stream', Buffer.from('MZ\x90\x00'));
  await expect(page.locator('.banner')).toContainText('Unsupported file type');
  await upload(page, 'fake.pdf', 'application/pdf', Buffer.from('this is plain text pretending to be a PDF'));
  await expect(page.locator('.banner')).toContainText('does not match its extension');
  await upload(page, 'empty.txt', 'text/plain', Buffer.from(''));
  await expect(page.locator('.banner')).toContainText('The file is empty');
  // Passes the upload checks (PDF signature) but cannot be parsed: processed, then marked failed.
  await upload(page, 'broken.pdf', 'application/pdf', Buffer.from('%PDF-1.4\nthis is not really a pdf body\n%%EOF'));
  const row = page.locator('.doc-row', { hasText: 'broken.pdf' });
  await expect(row.locator('.status-failed')).toBeVisible({ timeout: 60_000 });
  await expect(row.locator('.doc-error')).toContainText('damaged or password-protected');
  await expect(row.getByTitle('Retry processing')).toBeVisible();
  await assertNoLeaks(page, bodies);
});

test('ui: oversized upload', async ({ page }) => {
  const bodies = watchResponses(page);
  await signUp(page);
  await upload(page, 'huge.txt', 'text/plain', Buffer.alloc(26 * 1024 * 1024, 'a'));
  await expect(page.locator('.banner')).toContainText(/25 MB|too large/i, { timeout: 60_000 });
  await expect(page.locator('.banner')).not.toContainText('check your connection');
  await expect(page.locator('.doc-row', { hasText: 'huge.txt' })).toHaveCount(0);
  await assertNoLeaks(page, bodies);
});

test('ui: expired session', async ({ page, playwright }) => {
  const email = await signUp(page);
  await send(page, 'Say hello in five words.');
  await expect(page.locator('.message-wrapper')).toHaveCount(2);
  await expect(page.locator('.stream-status')).toHaveCount(0, { timeout: 120_000 });
  // The session is revoked elsewhere (e.g. "log out of all devices" on another computer).
  const other = await playwright.request.newContext({ baseURL: page.url() });
  const r = await other.post('/api/v1/auth/login', { data: { email, password: PASSWORD } });
  const token = (await r.json()).access_token;
  expect((await other.post('/api/v1/auth/logout-all', { headers: { Authorization: `Bearer ${token}` } })).status()).toBe(204);
  await other.dispose();
  // The next action finds the session gone: the app returns to the sign-in screen cleanly.
  await send(page, 'Are you still there?');
  await expect(page.getByText('Sign up with email')).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('body')).not.toContainText('REFRESH_TOKEN');
  // Signing in again works and the earlier conversation is still there.
  await page.getByRole('button', { name: 'Log in', exact: true }).click();
  await page.getByPlaceholder('Email address').fill(email);
  await page.getByPlaceholder('Password').fill(PASSWORD);
  await page.getByRole('button', { name: 'Log in', exact: true }).click();
  await expect(page.locator('.history-item')).toHaveCount(1);
});

test('ui: cancelled generation', async ({ page }) => {
  const bodies = watchResponses(page);
  await signUp(page);
  await send(page, 'Write a detailed 600-word essay about the history of bicycles.');
  const last = page.locator('.message-wrapper').last();
  await expect(last.locator('.markdown')).toBeVisible({ timeout: 120_000 });  // some text has streamed
  await page.getByTitle('Stop generating').click();
  await expect(last.locator('.message-error')).toContainText('Response stopped.');
  await expect(last.getByRole('button', { name: 'Retry' })).toBeVisible();
  const partial = (await last.locator('.markdown').textContent()) ?? '';
  expect(partial.length).toBeGreaterThan(0);
  // Persisted as stopped (with the partial text), not as an error or a finished answer.
  await page.reload();
  await page.locator('.history-item').first().click();
  const reloaded = page.locator('.message-wrapper').last();
  await expect(reloaded.locator('.message-error')).toContainText('Response stopped.');
  // Retry regenerates the answer.
  await reloaded.getByRole('button', { name: 'Retry' }).click();
  await expect(page.locator('.stream-status')).toHaveCount(0, { timeout: 180_000 });
  await expect(page.locator('.message-wrapper').last().locator('.message-error')).toHaveCount(0);
  await expect(page.locator('.message-wrapper').last().locator('.markdown')).toContainText(/bicycle/i);
  expect(((await page.locator('.message-wrapper').last().locator('.markdown').textContent()) ?? '').length).toBeGreaterThan(partial.length);
  await assertNoLeaks(page, bodies);
});
