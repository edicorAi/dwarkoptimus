import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// Injects a GoatCounter analytics tag into index.html only when the
// VITE_GOATCOUNTER_CODE env var is set at build time. Local dev runs
// (where the env is unset) render no script — your own visits don't
// pollute the dashboard. The code itself is public on the live site
// anyway; the env-var indirection just keeps it out of the repo and
// makes it easy to swap providers later.
function analyticsPlugin(): Plugin {
  return {
    name: "dwarkoptimus-analytics",
    transformIndexHtml(html) {
      const code = process.env.VITE_GOATCOUNTER_CODE?.trim();
      if (!code) return html;
      const tag = `<script data-goatcounter="https://${code}.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>`;
      return html.replace("</head>", `    ${tag}\n  </head>`);
    },
  };
}

export default defineConfig({
  plugins: [react(), analyticsPlugin()],
  base: process.env.GITHUB_PAGES === "true" ? "/dwarkoptimus/" : "/",
  server: {
    port: 5173,
  },
});
