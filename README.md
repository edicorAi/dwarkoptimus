# LLM Systems Calculator

React 19 + TypeScript + Vite app for estimating whether a selected LLM fits on the user's NVIDIA hardware (B300, H200, etc.). Produces a fit verdict, bottleneck explanation, and a suggested `vllm serve` command.

## What it does

Pick a model and a hardware preset. The app translates roofline math into plain operational guidance:

- **Can this model fit on this hardware** at the selected precision and context length?
- **Where will it bottleneck** — compute, weight-memory bandwidth, or KV-cache pressure?
- **What batch size** is needed to amortize the weight fetch, and what's the largest batch the HBM budget actually allows at the chosen context?
- **How do pipeline stages, expert parallelism, context length, and precision** change the verdict?
- **Which numbers are sourced and which are estimated** — every preset carries a confidence badge, and KV-bytes-per-token is flagged separately because it's almost always estimated.
- **What flags should I pass to vLLM** — the planning panel emits a ready-to-paste `vllm serve` command that respects the planned concurrency, context, and safety margin.

Sample verdicts the app is designed to produce:

- "This fits comfortably on your B300 server."
- "This is tight because KV cache dominates at 1M context."
- "Lowering context from 1M to 256K makes this configuration viable."
- "The selected model is an embedding model, so decode/KV estimates are not applicable."

## Why we built it

We operate a real, mixed NVIDIA inventory (a B300 8-GPU server, an H200 4-GPU server, and a 16-GPU H200 pool) and serve a curated set of models through LiteLLM. Before committing a model to that fleet we kept asking the same questions — *will it fit at the context we want, what's the bottleneck, what should the vLLM args be* — and answering them by hand on a whiteboard from model cards and GPU spec sheets.

This app is that whiteboard, made reproducible:

- Hardware and model presets are versioned in the repo, so the assumptions behind a verdict are auditable rather than tribal.
- Model preset names track the LiteLLM config at `/Users/ad/data/code/gitops/desls-lab/litellm/overlay/config.yaml`, so a verdict here maps directly to a model we actually serve.
- The calculation engine is a pure, unit-tested module — the same math drives the UI, the comparison table, and the vLLM-command generator, so the operator guidance can't drift from the underlying numbers.
- Confidence badges (`source-backed | estimated | user-provided | unknown`) make it obvious when a "fit" verdict rests on a guessed KV-cache size and should be validated with a real vLLM memory profile.

It is intentionally a static frontend — no backend, no auth, no saved state — because the goal is fast capacity-planning answers, not a long-lived dashboard.

## Local Development

```bash
pnpm install
pnpm run dev
```

Open:

```text
http://localhost:5173/
```

## Tests And Build

```bash
pnpm run test                 # vitest run
pnpm run build                # tsc typecheck, then vite build → dist/
pnpm run preview              # serve the built dist/
```

Run a single test:

```bash
pnpm exec vitest run -t "computes DeepSeek-style batch threshold"
pnpm exec vitest run src/lib/calculations.test.ts
```

## Docker

Build:

```bash
docker build -t llm-systems-calculator:local .
```

Run:

```bash
docker run --rm -p 8080:80 llm-systems-calculator:local
```

Open:

```text
http://localhost:8080/
```

## Project Layout

- `src/lib/calculations.ts` — pure roofline / memory-fit engine (no React). All math lives here and is tested directly.
- `src/data/hardware.ts`, `src/data/models.ts` — static hardware and model presets, each carrying a `confidence` badge.
- `src/main.tsx` — the entire React UI (Planner / Advanced / Settings tabs).
- `src/types.ts` — shared types.

## Notes

- Hardware presets include the user's NVIDIA B300 8-GPU server, one H200 4-GPU server, and the full 16-GPU H200 pool.
- Model presets match the LiteLLM model names in `/Users/ad/data/code/gitops/desls-lab/litellm/overlay/config.yaml`.
- KV bytes per token is often estimated because model cards rarely publish an exact runtime KV-cache footprint.
- `maxFittingBatch` (HBM-budget cap at the selected context) and `batchThreshold` (cost-optimal weight-amortization target) are deliberately separate metrics.
- The safety margin (default 0.8) is applied to per-GPU HBM and reused as vLLM's `--gpu-memory-utilization` in the suggested command.
