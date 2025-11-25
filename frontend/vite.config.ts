import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000, // We'll run the frontend on port 3000
    proxy: {
      // Proxy API requests to the Python backend server
      '/socket.io': {
        target: 'ws://127.0.0.1:8000',
        ws: true
      }
    }
  }
})
