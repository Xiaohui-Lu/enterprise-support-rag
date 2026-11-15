/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run build` writes dist/, which `srag serve` serves at / (D-051). During development,
// `npm run dev` serves the page itself and passes /api on to the service.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { environment: "node" },
});
