import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite dev server. The Python backend listens on 127.0.0.1:8001;
// we proxy /socket.io (HTTP + WebSocket) to it.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 3000,
    strictPort: true,
    proxy: {
      '/socket.io': {
        target: 'http://127.0.0.1:8001',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
