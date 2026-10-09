import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: {
      // Same-origin in development: the refresh cookie and SSE work without CORS.
      proxy: {
        '/api': { target: env.VITE_API_PROXY_TARGET || 'http://localhost:8000', changeOrigin: false },
      },
    },
  }
})
