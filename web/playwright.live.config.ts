import { defineConfig } from '@playwright/test';
import base from './playwright.config';

// The same end-to-end suite, run against the deployed GitHub Pages site:
//   PW_CHROMIUM=... npx playwright test -c playwright.live.config.ts
// LIVE_URL overrides the target (it must end with a slash).
export default defineConfig({
  ...base,
  retries: 1,
  use: { ...base.use, baseURL: process.env.LIVE_URL ?? 'https://kyleyhw.github.io/sound_simulation/' },
  webServer: undefined,
});
