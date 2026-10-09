import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end tests against the full stack started by scripts/run_stack.py --mode e2e:
 * frontend -> backend -> agent-core -> rag-core / llm -> mock FreeLLMAPI (no real keys needed).
 *
 *   npm run test:e2e
 *
 * Uses the locally installed Chrome (channel "chrome"); set PW_CHANNEL="" to use Playwright's
 * bundled Chromium instead (after `npx playwright install chromium`).
 */
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PYTHON = process.env.PYTHON ?? path.join(ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const channel = process.env.PW_CHANNEL ?? 'chrome';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:5174',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...(channel ? { channel } : {}),
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], ...(channel ? { channel } : {}) }, testIgnore: /mobile.spec.ts/ },
    { name: 'mobile', use: { ...devices['Pixel 7'], ...(channel ? { channel } : {}) }, testMatch: /mobile\.spec\.ts/ },
  ],
  webServer: {
    command: `"${PYTHON}" "${path.join(ROOT, 'scripts', 'run_stack.py')}" --mode e2e`,
    url: 'http://localhost:5174',
    timeout: 180_000,
    reuseExistingServer: false,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
