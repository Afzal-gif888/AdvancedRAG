import { expect, type Page } from '@playwright/test';

export const PASSWORD = 'E2e-password-1';

export function uniqueEmail(prefix = 'e2e'): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1e6)}@example.com`;
}

/** Sign-up (name, email, password) lands directly in the app. */
export async function register(page: Page, email = uniqueEmail(), name = 'Ada Lovelace'): Promise<string> {
  await page.goto('/');
  await page.getByText('Sign up with email').click();
  await page.getByPlaceholder('Full name').fill(name);
  await page.getByPlaceholder('Email address').fill(email);
  await page.getByPlaceholder('Password').fill(PASSWORD);
  await page.getByRole('button', { name: 'Create account' }).click();
  await expect(page.getByText('How can I help you today?')).toBeVisible();
  return email;
}

export async function send(page: Page, text: string): Promise<void> {
  await page.locator('.composer-textarea').fill(text);
  await page.locator('.composer-textarea').press('Enter');
}

export async function openDocuments(page: Page): Promise<void> {
  if (!(await page.locator('.context-panel').isVisible())) await page.getByTitle('Documents').click();
}
