import { expect, test } from '@playwright/test';
import { register, send } from './helpers';

test('mobile: chat, sidebar and documents panel are usable without horizontal scroll', async ({ page }) => {
  await register(page);
  await send(page, 'Mobile question with `inline code`');
  await expect(page.locator('.message-content .markdown').last()).toContainText('Mock model reply');

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);

  await page.locator('.menu-toggle').click();
  await expect(page.locator('.sidebar.open')).toBeVisible();
  await expect(page.locator('.history-item').first()).toBeVisible();
  await page.locator('.history-item').first().click();

  // Knowledge Context opens inside the sidebar, below the icons: the menu stays open to show it.
  await page.locator('.menu-toggle').click();
  await page.locator('.sidebar-header').getByTitle('Documents').click();
  await expect(page.locator('.sidebar.open .context-panel .upload-area')).toBeVisible();
  // No collapse arrow: tapping outside the menu closes it.
  await page.mouse.click(370, 300);
  await expect(page.locator('.sidebar.open')).toHaveCount(0);
  const overflow2 = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow2).toBeLessThanOrEqual(1);
});
