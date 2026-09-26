import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';
import { installAnchorLinks } from './lib/anchors';
import { useApp } from './state/store';

import { PRESETS } from './engine/presets';
import type { Scene } from './engine/scene';

// Exposed for end-to-end tests and console exploration.
(window as unknown as { __app: typeof useApp }).__app = useApp;
Object.assign(window, {
  __presets: PRESETS,
  /** GPU-vs-CPU parity for a scene (loads the WebGPU code on demand). */
  __gpuParity: async (scene: Scene, steps: number, batches?: number) => (await import('./engine/gpuParity')).gpuParity(scene, steps, batches),
});

// In-page #anchor links in rendered Markdown scroll instead of routing.
installAnchorLinks();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
