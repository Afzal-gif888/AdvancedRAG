import { expect, test } from '@playwright/test';
import { PASSWORD, openDocuments, register, send, uniqueEmail } from './helpers';

test('sign-up signs in directly; login is email + password only', async ({ page }) => {
  const email = uniqueEmail();
  await page.goto('/');
  await page.getByText('Sign up with email').click();
  // Name, email and password are all required.
  await expect(page.getByRole('button', { name: 'Create account' })).toBeDisabled();
  await page.getByPlaceholder('Full name').fill('Grace Hopper');
  await page.getByPlaceholder('Email address').fill(email);
  await page.getByPlaceholder('Password').fill('weak');
  await page.getByRole('button', { name: 'Create account' }).click();
  await expect(page.locator('.form-error')).toContainText('at least 8 characters');

  await page.getByPlaceholder('Password').fill(PASSWORD);
  await page.getByRole('button', { name: 'Create account' }).click();
  // No code step: the new account goes straight into the app.
  await expect(page.getByText('How can I help you today?')).toBeVisible();

  await page.locator('.sidebar-footer').click();
  await page.getByText('Log out').click();
  await expect(page.getByText('Sign up with email')).toBeVisible();

  await page.getByRole('button', { name: 'Log in', exact: true }).click();
  await page.getByPlaceholder('Email address').fill(email);
  await page.getByPlaceholder('Password').fill('Wrong-password-9');
  await page.getByRole('button', { name: 'Log in', exact: true }).click();
  await expect(page.locator('.form-error')).toHaveText('Invalid email or password.');
  await page.getByPlaceholder('Password').fill(PASSWORD);
  await page.getByRole('button', { name: 'Log in', exact: true }).click();
  await expect(page.getByText('How can I help you today?')).toBeVisible();

  // The same email cannot be registered twice.
  await page.locator('.sidebar-footer').click();
  await page.getByText('Log out').click();
  await page.getByText('Sign up with email').click();
  await page.getByPlaceholder('Full name').fill('Someone Else');
  await page.getByPlaceholder('Email address').fill(email);
  await page.getByPlaceholder('Password').fill('Other-password-2');
  await page.getByRole('button', { name: 'Create account' }).click();
  await expect(page.locator('.form-error')).toContainText('already exists');
});

test('profile has no change-password option', async ({ page }) => {
  await register(page);
  await page.locator('.sidebar-footer').click();
  await page.getByText('Profile').click();
  await expect(page.getByRole('dialog', { name: 'Profile' })).toBeVisible();
  await expect(page.getByText(/change password/i)).toHaveCount(0);
  await expect(page.locator('.modal-content input[type=password]')).toHaveCount(0);
});

test('chat streams a markdown answer, persists it, and survives reload', async ({ page }) => {
  await register(page);
  await send(page, 'Explain hybrid retrieval');
  const answer = page.locator('.message-content .markdown').last();
  await expect(answer).toContainText('Mock model reply');
  // Markdown rendered as elements, not raw text.
  await expect(answer.locator('strong')).toHaveText('Mock model reply');
  await expect(answer.locator('li')).toHaveCount(2);

  const title = page.locator('.history-item.active .history-item-title');
  await expect(title).toHaveText('Explain hybrid retrieval');
  await expect(page.locator('.history-item.active .history-item-time')).toHaveText('now');

  await page.reload();
  await page.locator('.history-item').first().click();
  await expect(page.locator('.message-content .markdown').last()).toContainText('Mock model reply');
  await expect(page.locator('.message-content').first()).toContainText('Explain hybrid retrieval');
});

test('Knowledge Base upload is answered with citations in every new chat', async ({ page }) => {
  await register(page);
  await send(page, 'hello');
  await expect(page.locator('.message-content .markdown').last()).toContainText('Mock model reply');

  await openDocuments(page);
  await page.locator('.upload-area input[type=file]').setInputFiles({
    name: 'warranty.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# Warranty\n\nEvery AdvancedWidget ships with a 24 month warranty covering defects.\n\nWater damage is excluded.'),
  });
  const row = page.locator('.doc-row', { hasText: 'warranty.md' });
  await expect(row.locator('.status-ready')).toBeVisible({ timeout: 30_000 });
  // Nothing to attach: no per-chat checkboxes, ready documents are searched in every chat.
  await expect(row.locator('input[type=checkbox]')).toHaveCount(0);
  await expect(page.locator('.panel-note')).toContainText('every chat');

  await send(page, 'How long is the warranty?');
  const answer = page.locator('.message-content').last();
  await expect(answer).toContainText('24 month warranty');
  await expect(answer.locator('.citations')).toContainText('warranty.md');
  await expect(answer.locator('.code-block')).toContainText('grounded=true');
  // The [1] marker in the text is a numbered badge linked to source 1, which opens its excerpt.
  await expect(answer.locator('.markdown')).not.toContainText('[1]');
  await answer.getByRole('button', { name: 'Source 1: warranty.md' }).click();
  await expect(answer.locator('.source-preview')).toContainText('24 month warranty');
  await expect(answer.locator('.source-chip.active .source-num')).toHaveText('1');

  // A brand-new chat uses the same Knowledge Base without re-uploading or attaching.
  await page.getByTitle('New chat').click();
  await expect(page.getByText('How can I help you today?')).toBeVisible();
  await send(page, 'Is water damage covered by the warranty?');
  const second = page.locator('.message-content').last();
  await expect(second.locator('.citations')).toContainText('warranty.md', { timeout: 30_000 });

  // Not in the Knowledge Base: the exact "insufficient evidence" answer, no citations.
  await send(page, 'What does our dental plan cover for employees?');
  const third = page.locator('.message-content').last();
  await expect(third).toContainText("I couldn't find sufficient evidence for this answer in your Knowledge Base.", { timeout: 30_000 });
  await expect(third.locator('.citations')).toHaveCount(0);
});

test('chats have no upload: files dropped on a chat are not uploaded and point to Knowledge Context', async ({ page }) => {
  await register(page);
  await expect(page.locator('.composer-wrapper input[type=file]')).toHaveCount(0);
  await expect(page.locator('.composer-wrapper').getByTitle('Attach')).toHaveCount(0);
  const dataTransfer = await page.evaluateHandle(() => {
    const dt = new DataTransfer();
    dt.items.add(new File(['dropped on the chat'], 'chat-drop.txt', { type: 'text/plain' }));
    return dt;
  });
  await page.locator('.main-chat').dispatchEvent('dragover', { dataTransfer });
  await page.locator('.main-chat').dispatchEvent('drop', { dataTransfer });
  await expect(page.locator('.banner')).toContainText('Knowledge Context panel');
  await expect(page.locator('.context-panel')).toBeVisible();
  await page.waitForTimeout(500);
  await expect(page.locator('.doc-row', { hasText: 'chat-drop.txt' })).toHaveCount(0);
});

test('rejects unsupported files with a clear error', async ({ page }) => {
  await register(page);
  await openDocuments(page);
  await page.locator('.upload-area input[type=file]').setInputFiles({ name: 'evil.exe', mimeType: 'application/octet-stream', buffer: Buffer.from('MZ') });
  await expect(page.locator('.banner')).toContainText('Unsupported file type');
});

test('search finds chats by message content and surfaces errors', async ({ page }) => {
  await register(page);
  await send(page, 'Tell me about pelicans');
  await expect(page.locator('.message-content .markdown').last()).toBeVisible();
  // One generation at a time: wait for the stream to finish before sending in another chat.
  await expect(page.getByTitle('Stop generating')).toHaveCount(0);
  await page.getByTitle('New chat').click();
  await send(page, 'Something unrelated');
  await expect(page.locator('.message-content .markdown').last()).toBeVisible();
  await expect(page.getByTitle('Stop generating')).toHaveCount(0);

  await page.getByTitle('Search chats').click();
  await page.getByPlaceholder('Search chats...').fill('pelicans');
  const results = page.locator('.modal-content .history-item');
  await expect(results).toHaveCount(1);
  await results.first().click();
  await expect(page.locator('.message-content').first()).toContainText('pelicans');

  // Backend failure is shown, not swallowed.
  await page.route('**/api/v1/search/chats**', (route) =>
    route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ error: { code: 'INTERNAL_SERVER_ERROR', message: 'An unexpected error occurred.' } }) }),
  );
  await page.getByTitle('Search chats').click();
  await page.getByPlaceholder('Search chats...').fill('anything');
  await expect(page.locator('.modal-content [role=alert]')).toHaveText('An unexpected error occurred.');
});

test('share link shows a read-only snapshot and can be revoked', async ({ page, browser }) => {
  await register(page);
  await send(page, 'Shareable question');
  await expect(page.locator('.message-content .markdown').last()).toContainText('Mock model reply');
  // Each chat in the list has its own Share button, next to Rename and Delete.
  const item = page.locator('.history-item', { hasText: 'Shareable question' });
  await item.hover();
  await item.getByTitle('Share').click();
  await page.getByRole('button', { name: /Create link/ }).click();
  const url = await page.locator('.modal-content input[readonly]').inputValue();
  expect(url).toMatch(/\/share\/[\w-]{40,}$/);

  const anon = await (await browser.newContext()).newPage();
  await anon.goto(url);
  await expect(anon.getByText('read-only')).toBeVisible();
  await expect(anon.locator('.message-wrapper')).toHaveCount(2);

  await page.getByRole('button', { name: 'Revoke link' }).click();
  await anon.reload();
  await expect(anon.locator('.form-error')).toContainText('no longer available');
});

test('settings: theme persists across reloads (server + local)', async ({ page }) => {
  await register(page);
  await page.locator('.sidebar-footer').click();
  await page.getByText('Settings').click();
  await page.locator('select').selectOption('light');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(255, 255, 255)');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.locator('.sidebar-footer').click();
  await page.getByText('Settings').click();
  await page.locator('select').selectOption('dark');
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(23, 23, 23)');
});

test('drag and drop uploads a document with progress', async ({ page }) => {
  await register(page);
  await openDocuments(page);
  const dataTransfer = await page.evaluateHandle(() => {
    const dt = new DataTransfer();
    dt.items.add(new File(['dropped file about kiwis'], 'dropped.txt', { type: 'text/plain' }));
    return dt;
  });
  await page.locator('.context-panel').dispatchEvent('dragover', { dataTransfer });
  await expect(page.locator('.context-panel')).toHaveClass(/drag-active/);
  await page.locator('.context-panel').dispatchEvent('drop', { dataTransfer });
  await expect(page.locator('.doc-row', { hasText: 'dropped.txt' }).locator('.status-ready')).toBeVisible({ timeout: 30_000 });
});

test('no secrets or tokens are exposed to the browser', async ({ page }) => {
  await register(page);
  const storage = await page.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }));
  expect(storage).not.toMatch(/eyJ|token|FREELLMAPI|api[_-]?key/i);
  expect(await page.evaluate(() => document.cookie)).toBe(''); // refresh cookie is HttpOnly
  const html = await page.content();
  expect(html).not.toMatch(/FREELLMAPI|e2e-mock-key|internal-service-token/i);
});

test('composer grows with long text, wraps unbroken text, scrolls past its max and shrinks after sending', async ({ page }) => {
  await register(page);
  const box = page.locator('.composer-textarea');
  const one = (await box.boundingBox())!.height;
  await box.fill(Array.from({ length: 5 }, (_, i) => `line ${i + 1}`).join('\n'));
  const five = (await box.boundingBox())!.height;
  expect(five).toBeGreaterThan(one * 3);
  await box.fill("'".repeat(600));  // one unbroken run of characters
  expect(await box.evaluate((el) => el.scrollWidth <= el.clientWidth + 1)).toBe(true);  // wrapped, not hidden sideways
  expect((await box.boundingBox())!.height).toBeGreaterThan(one * 2);
  await box.fill(Array.from({ length: 40 }, (_, i) => `line ${i + 1}`).join('\n'));
  const tall = (await box.boundingBox())!.height;
  expect(tall).toBeLessThanOrEqual(201);  // capped at max-height...
  expect(await box.evaluate((el) => getComputedStyle(el).overflowY)).toBe('auto');  // ...and scrollable
  await box.press('Enter');
  await expect(box).toHaveValue('');
  await expect.poll(async () => (await box.boundingBox())!.height).toBeLessThanOrEqual(one + 1);
});

test('layout: new chat, search and documents beside each other at the top; share/rename/delete on every chat', async ({ page }) => {
  await register(page);
  await send(page, 'hello');
  await expect(page.locator('.history-item').first()).toBeVisible();
  await expect(page.getByTitle('Close sidebar')).toHaveCount(0);  // no collapse arrow
  const header = page.locator('.sidebar-header .sidebar-header-actions');
  for (const title of ['New chat', 'Search chats', 'Documents']) await expect(header.getByTitle(title)).toBeVisible();
  for (const title of ['Search chats', 'Share chat', 'Documents']) await expect(page.locator('.chat-header').getByTitle(title)).toHaveCount(0);
  const y = async (title: string) => (await header.getByTitle(title).boundingBox())!.y;
  expect(Math.abs((await y('Search chats')) - (await y('New chat')))).toBeLessThan(2);  // same row
  expect(Math.abs((await y('Documents')) - (await y('New chat')))).toBeLessThan(2);

  // Every new chat gets Share alongside Rename and Delete.
  await page.getByTitle('New chat').click();
  await send(page, 'second chat');
  await expect(page.locator('.history-item')).toHaveCount(2);
  for (const item of await page.locator('.history-item').all()) {
    await item.hover();
    for (const title of ['Share', 'Rename', 'Delete']) await expect(item.locator('.history-item-actions').getByTitle(title)).toBeVisible();
  }

  // Knowledge Context still opens below the header, chats below it; upload works as before.
  const box = async (sel: string) => (await page.locator(sel).first().boundingBox())!;
  await header.getByTitle('Documents').click();
  const head = await box('.sidebar-header');
  const panel = await box('.sidebar .context-panel');
  expect(panel.y).toBeGreaterThanOrEqual(head.y + head.height - 1);
  expect((await box('.chat-history')).y).toBeGreaterThanOrEqual(panel.y + panel.height - 1);
  await page.locator('.upload-area input[type=file]').setInputFiles({ name: 'sidebar-panel.txt', mimeType: 'text/plain', buffer: Buffer.from('Upload from the sidebar works.') });
  await expect(page.locator('.doc-row', { hasText: 'sidebar-panel.txt' }).locator('.status-ready')).toBeVisible({ timeout: 30_000 });
  await header.getByTitle('Documents').click();
  await expect(page.locator('.context-panel')).toHaveCount(0);
});

test('deleting a chat asks in an in-app dialog (no browser pop-up)', async ({ page }) => {
  let nativeDialog = false;
  page.on('dialog', async (d) => { nativeDialog = true; await d.dismiss(); });
  await register(page);
  await send(page, 'Chat to delete');
  const item = page.locator('.history-item', { hasText: 'Chat to delete' });
  await expect(item).toBeVisible();
  const open = async () => { await item.hover(); await item.getByTitle('Delete').click(); };
  const dialog = page.getByRole('alertdialog', { name: 'Delete chat?' });

  await open();
  await expect(dialog).toContainText('This will delete Chat to delete.');
  await dialog.getByRole('button', { name: 'Cancel' }).click();
  await expect(dialog).toHaveCount(0);
  await expect(item).toBeVisible();

  await open();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(item).toBeVisible();

  await open();
  await dialog.getByRole('button', { name: 'Delete' }).click();
  await expect(dialog).toHaveCount(0);
  await expect(item).toHaveCount(0);
  await page.reload();
  await expect(page.locator('.history-item', { hasText: 'Chat to delete' })).toHaveCount(0);  // deleted on the server
  expect(nativeDialog).toBe(false);
});

test('every destructive action uses the in-app dialog: document delete, clear all data, log out everywhere', async ({ page }) => {
  let nativeDialog = false;
  page.on('dialog', async (d) => { nativeDialog = true; await d.dismiss(); });
  await register(page);

  // Knowledge Base document
  await openDocuments(page);
  await page.locator('.upload-area input[type=file]').setInputFiles({ name: 'to-delete.txt', mimeType: 'text/plain', buffer: Buffer.from('Delete me.') });
  const row = page.locator('.doc-row', { hasText: 'to-delete.txt' });
  await expect(row.locator('.status-ready')).toBeVisible({ timeout: 30_000 });
  const docDialog = page.getByRole('alertdialog', { name: 'Delete document?' });
  await row.getByTitle('Delete', { exact: true }).click();
  await expect(docDialog).toContainText('This will delete to-delete.txt from your Knowledge Base.');
  await docDialog.getByRole('button', { name: 'Cancel' }).click();
  await expect(row).toBeVisible();
  await row.getByTitle('Delete', { exact: true }).click();
  await docDialog.getByRole('button', { name: 'Delete' }).click();
  await expect(docDialog).toHaveCount(0);
  await expect(row).toHaveCount(0);
  await page.reload();
  await openDocuments(page);
  await expect(page.locator('.doc-row', { hasText: 'to-delete.txt' })).toHaveCount(0);  // deleted on the server

  // Settings: Clear All Data (cancel, then confirm)
  await send(page, 'a chat that will be cleared');
  await expect(page.locator('.history-item')).toHaveCount(1);
  await page.locator('.sidebar-footer').click();
  await page.getByText('Settings').click();
  const clear = page.getByRole('alertdialog', { name: 'Clear all data?' });
  await page.getByRole('button', { name: 'Clear All Data' }).click();
  await clear.getByRole('button', { name: 'Cancel' }).click();
  await expect(clear).toHaveCount(0);
  await expect(page.getByRole('dialog', { name: 'Settings' })).toBeVisible();  // settings stay open
  await page.getByRole('button', { name: 'Clear All Data' }).click();
  await clear.getByRole('button', { name: 'Delete everything' }).click();
  await expect(page.locator('.history-item')).toHaveCount(0);

  // Settings: Log out of all devices
  await page.locator('.sidebar-footer').click();
  await page.getByText('Settings').click();
  const logout = page.getByRole('alertdialog', { name: 'Log out of all devices?' });
  await page.getByRole('button', { name: 'Log out of all devices' }).click();
  await page.keyboard.press('Escape');
  await expect(logout).toHaveCount(0);
  await page.getByRole('button', { name: 'Log out of all devices' }).click();
  await logout.getByRole('button', { name: 'Log out' }).click();
  await expect(page.getByText('Sign up with email')).toBeVisible();
  expect(nativeDialog).toBe(false);
});

test('a first visit makes no failing request, and every form field is named for autofill', async ({ page }) => {
  const failed: string[] = [];
  const consoleErrors: string[] = [];
  let refreshCalls = 0;
  page.on('response', (r) => { if (r.status() >= 400) failed.push(`${r.status()} ${new URL(r.url()).pathname}`); });
  page.on('request', (r) => { if (r.url().includes('/auth/refresh')) refreshCalls++; });
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });

  await page.goto('/');
  await expect(page.getByText('Sign up with email')).toBeVisible();
  // Never signed in on this browser: no cookie to try, so no refresh call (it would be a guaranteed 401).
  expect(refreshCalls).toBe(0);

  for (const open of ['Sign up with email', 'Log in']) {
    await page.goto('/');
    await page.getByRole('button', { name: open, exact: true }).click();
    const unnamed = await page.locator('input:visible, textarea:visible').evaluateAll(
      (els) => els.filter((e) => !(e as HTMLInputElement).id && !(e as HTMLInputElement).name).length,
    );
    expect(unnamed, `inputs without id or name on "${open}"`).toBe(0);
  }
  expect(failed).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test('a signed-in browser still restores its session, and logging out forgets it', async ({ page }) => {
  await register(page);
  let refreshCalls = 0;
  page.on('request', (r) => { if (r.url().includes('/auth/refresh')) refreshCalls++; });

  await page.reload();
  await expect(page.getByText('How can I help you today?')).toBeVisible(); // restored through the cookie
  expect(refreshCalls).toBe(1);

  await page.locator('.sidebar-footer').click();
  await page.getByText('Log out').click();
  await expect(page.getByText('Sign up with email')).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem('advancedrag.signed-in'))).toBeNull();

  refreshCalls = 0;
  await page.reload();
  await expect(page.getByText('Sign up with email')).toBeVisible();
  expect(refreshCalls).toBe(0); // forgotten: no needless 401 after a sign-out either
});
