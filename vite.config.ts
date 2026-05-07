import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// Injects a GoatCounter analytics tag into index.html only when the
// VITE_GOATCOUNTER_CODE env var is set at build time. Local dev runs
// (where the env is unset) render no script — your own visits don't
// pollute the dashboard. The code itself is public on the live site
// anyway; the env-var indirection just keeps it out of the repo and
// makes it easy to swap providers later.
//
// count.js is fetched from gc.zgo.at at build time and emitted as a
// same-origin asset. Loading it directly from gc.zgo.at causes a
// large fraction of real visitor pings to be silently dropped — that
// host is on essentially every ad-blocker and tracker-DNS blocklist
// (uBlock Origin, NextDNS, StevenBlack hosts, etc.).
function analyticsPlugin(): Plugin {
  let base = "/";
  return {
    name: "dwarkoptimus-analytics",
    configResolved(config) {
      base = config.base;
    },
    async buildStart() {
      const code = process.env.VITE_GOATCOUNTER_CODE?.trim();
      if (!code) return;
      const res = await fetch("https://gc.zgo.at/count.js");
      if (!res.ok) {
        throw new Error(`Failed to fetch GoatCounter count.js: HTTP ${res.status}`);
      }
      this.emitFile({
        type: "asset",
        fileName: "gc-count.js",
        source: await res.text(),
      });
    },
    transformIndexHtml(html) {
      const code = process.env.VITE_GOATCOUNTER_CODE?.trim();
      if (!code) return html;
      const tag = `<script data-goatcounter="https://${code}.goatcounter.com/count" async src="${base}gc-count.js"></script>`;
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
