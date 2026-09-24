import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// BASE is set by the Pages workflow to "/<repo>/"; local dev serves at "/".
export default defineConfig({
  base: process.env.BASE ?? '/',
  plugins: [react()],
  server: { host: '127.0.0.1', port: 3000 },
  build: { target: 'es2022', sourcemap: true, chunkSizeWarningLimit: 2000 },
  test: {
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
  },
});
