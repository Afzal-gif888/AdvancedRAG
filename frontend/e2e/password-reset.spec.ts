import { expect, test, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PASSWORD, register } from './helpers';

// The e2e stack runs the backend with the development "file" email provider: every email is
// written to .e2e-outbox (see scripts/run_stack.py), so the real reset link can be followed.
const OUTBOX = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '.e2e-outbox');
const NEW_PASSWORD = 'Reset-password-2';
const GENERIC = 'If an account exists for this email, a password reset link has been sent.';

async function resetLinkFor(email: string): Promise<string> {
  let link: string | undefined;
  await expect.poll(() => {
    const files = fs.existsSync(OUTBOX) ? fs.readdirSync(OUTBOX).sort() : [];
    for (const f of files.reverse()) {
      const text = fs.readFileSync(path.join(OUTBOX, f), 'utf-8');
      if (text.startsWith(`To: ${email}\n`)) {
        link = text.match(/https?:\/\/\S+\/reset-password\?token=[A-Za-z0-9_-]+/)?.[0];
        if (link) return true;
      }
    }
    return false;
  }, { timeout: 10_000 }).toBe(true);
  return new URL(link!).pathname + new URL(link!).search;
}

async function requestReset(page: Page, email: string) {
  await page.goto('/forgot-password');
  await page.getByPlaceholder('Email address').fill(email);
  await page.getByRole('button', { name: 'Send reset link' }).click();
  await expect(page.getByText(GENERIC)).toBeVisible();
}

async function signIn(page: Page, email: string, password: string) {
  await page.goto('/');
  await page.getByRole('button', { name: 'Log in' }).click();
  await page.getByPlaceholder('Email address').fill(email);
  await page.getByPlaceholder('Password').fill(password);
  await page.getByRole('button', { name: 'Log in' }).click();
}

test('forgot-password page is reachable from login and shows the generic message', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Log in' }).click();
  await page.getByPlaceholder('Email address').fill('typed@example.com');
  await page.getByRole('link', { name: 'Forgot password?' }).click();
  await expect(page).toHaveURL(/\/forgot-password$/);
  await expect(page.getByPlaceholder('Email address')).toHaveValue('typed@example.com');  // carried over
  await page.getByRole('button', { name: 'Send reset link' }).click();
  // Same message whether or not the account exists.
  await expect(page.getByText(GENERIC)).toBeVisible();
  await page.getByRole('link', { name: 'Back to sign in' }).click();
  await expect(page.getByText('Sign up with email')).toBeVisible();
});

test('full reset: email link -> new password -> old fails, new works, link is single-use', async ({ page }) => {
  const email = await register(page);
  await page.context().clearCookies();
  await requestReset(page, email);
  const link = await resetLinkFor(email);

  await page.goto(link);
  await expect(page.getByText('Choose a new password')).toBeVisible();
  // Client-side validation before anything is sent.
  await page.getByPlaceholder('New password', { exact: true }).fill(NEW_PASSWORD);
  await page.getByPlaceholder('Confirm new password').fill('Different-pass-3');
  await page.getByRole('button', { name: 'Reset password' }).click();
  await expect(page.getByRole('alert')).toHaveText('Passwords do not match.');
  await page.getByPlaceholder('New password', { exact: true }).fill('nodigits');
  await page.getByPlaceholder('Confirm new password').fill('nodigits');
  await page.getByRole('button', { name: 'Reset password' }).click();
  await expect(page.getByRole('alert')).toHaveText('Password must contain at least one letter and one number.');

  await page.getByPlaceholder('New password', { exact: true }).fill(NEW_PASSWORD);
  await page.getByPlaceholder('Confirm new password').fill(NEW_PASSWORD);
  await page.getByRole('button', { name: 'Reset password' }).click();
  await expect(page.getByText('Your password has been reset. Sign in with your new password.')).toBeVisible();

  await signIn(page, email, PASSWORD);
  await expect(page.getByRole('alert')).toBeVisible();  // old password rejected
  await signIn(page, email, NEW_PASSWORD);
  await expect(page.getByText('How can I help you today?')).toBeVisible();

  // The same link cannot be used again.
  await page.context().clearCookies();
  await page.goto(link);
  await page.getByPlaceholder('New password', { exact: true }).fill('Third-password-4');
  await page.getByPlaceholder('Confirm new password').fill('Third-password-4');
  await page.getByRole('button', { name: 'Reset password' }).click();
  await expect(page.getByText('This reset link is invalid or has expired.')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Request a new link' })).toHaveAttribute('href', '/forgot-password');
});

test('malformed, missing and unknown tokens show the invalid-link state', async ({ page }) => {
  for (const url of ['/reset-password', '/reset-password?token=short', '/reset-password?token=%3Cscript%3E']) {
    await page.goto(url);
    await expect(page.getByText('This reset link is invalid or has expired.')).toBeVisible();
    await expect(page.getByPlaceholder('New password', { exact: true })).toHaveCount(0);
  }
  // Well-formed but never issued: the server answers INVALID_TOKEN.
  await page.goto(`/reset-password?token=${'A'.repeat(43)}`);
  await page.getByPlaceholder('New password', { exact: true }).fill(NEW_PASSWORD);
  await page.getByPlaceholder('Confirm new password').fill(NEW_PASSWORD);
  await page.getByRole('button', { name: 'Reset password' }).click();
  await expect(page.getByText('This reset link is invalid or has expired.')).toBeVisible();
});

test('loading states are shown while requests are in flight', async ({ page }) => {
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  await page.route('**/api/v1/auth/forgot-password', async (route) => { await gate; await route.continue(); });
  await page.goto('/forgot-password');
  await page.getByPlaceholder('Email address').fill('slow@example.com');
  await page.getByRole('button', { name: 'Send reset link' }).click();
  await expect(page.getByRole('button', { name: 'Sending…' })).toBeDisabled();
  release();
  await expect(page.getByText(GENERIC)).toBeVisible();

  let releaseReset!: () => void;
  const resetGate = new Promise<void>((r) => { releaseReset = r; });
  await page.route('**/api/v1/auth/reset-password', async (route) => { await resetGate; await route.continue(); });
  await page.goto(`/reset-password?token=${'B'.repeat(43)}`);
  await page.getByPlaceholder('New password', { exact: true }).fill(NEW_PASSWORD);
  await page.getByPlaceholder('Confirm new password').fill(NEW_PASSWORD);
  await page.getByRole('button', { name: 'Reset password' }).click();
  await expect(page.getByRole('button', { name: 'Saving…' })).toBeDisabled();
  releaseReset();
  await expect(page.getByText('This reset link is invalid or has expired.')).toBeVisible();
});

test('server and network errors are shown safely', async ({ page }) => {
  const internal = 'psycopg.OperationalError: connection to 10.0.0.5 failed; RESEND key re_live_123';
  await page.route('**/api/v1/auth/forgot-password', (route) =>
    route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ error: { code: 'INTERNAL_SERVER_ERROR', message: internal } }) }));
  await page.goto('/forgot-password');
  await page.getByPlaceholder('Email address').fill('err@example.com');
  await page.getByRole('button', { name: 'Send reset link' }).click();
  await expect(page.getByRole('alert')).toHaveText('Something went wrong. Please try again in a moment.');
  await expect(page.getByText(internal)).toHaveCount(0);

  await page.route('**/api/v1/auth/reset-password', (route) => route.abort('failed'));
  await page.goto(`/reset-password?token=${'C'.repeat(43)}`);
  await page.getByPlaceholder('New password', { exact: true }).fill(NEW_PASSWORD);
  await page.getByPlaceholder('Confirm new password').fill(NEW_PASSWORD);
  await page.getByRole('button', { name: 'Reset password' }).click();
  await expect(page.getByRole('alert')).toHaveText('Network error. Please check your connection.');
  // Rate limiting is surfaced with the API's own (safe) message.
  await page.unroute('**/api/v1/auth/reset-password');
  await page.route('**/api/v1/auth/reset-password', (route) =>
    route.fulfill({ status: 429, contentType: 'application/json', headers: { 'retry-after': '60' }, body: JSON.stringify({ error: { code: 'RATE_LIMITED', message: 'Too many requests. Please try again later.' } }) }));
  await page.getByRole('button', { name: 'Reset password' }).click();
  await expect(page.getByRole('alert')).toContainText('Too many requests');
});
