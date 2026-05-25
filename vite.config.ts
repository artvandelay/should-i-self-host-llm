import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves at https://<user>.github.io/<repo>/, so assets need this prefix.
  base: "/should-i-self-host-llm/",
});