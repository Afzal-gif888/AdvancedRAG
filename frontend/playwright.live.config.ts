import { defineConfig, devices } from '@playwright/test';

/**
 * LIVE UI verification against the already-running development stack (no servers are started):
 *   frontend :5173 -> backend :8000 -> Agent Core -> RAG Core (bge-small + Pinecone) -> LLM service -> FreeLLMAPI
 * Real models answer, so assertions check facts/citations, not exact wording.
 *
 *   npx playwright test -c playwright.live.config.ts
 */
const channel = process.env.PW_CHANNEL ?? 'chrome';

export default defineConfig({
  testDir: './e2e-live',
  timeout: 240_000,
  expect: { timeout: 90_000 },
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.LIVE_BASE_URL ?? 'http://localhost:5173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...(channel ? { channel } : {}),
  },
  projects: [{ name: 'live', use: { ...devices['Desktop Chrome'], ...(channel ? { channel } : {}) } }],
});
