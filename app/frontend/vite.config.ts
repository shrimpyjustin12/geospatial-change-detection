import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies /api to the FastAPI backend on :7860 so `npm run dev` works against it.
// In production the same FastAPI process serves this build as static files (single origin).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:7860",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
