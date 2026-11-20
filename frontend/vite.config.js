import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy all /api/* requests to Django backend in dev to avoid CORS issues
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy media files served by Django
      '/media': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
