import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the gateway runs on :8501 and serves /api and the sign-in routes.
// Run `VITE_MOCK=1 npm run dev` to preview the UI with sample data and no backend.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8501",
      "/auth": "http://localhost:8501",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
