import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { hardwarePresets } from "./data/hardware";
import { modelPresets } from "./data/models";
import {
  autoOptimize,
  buildScenario,
  calculateScenario,
  createServingPlan,
  getRooflineSweep,
} from "./lib/calculations";
import { buildScenarioReport } from "./lib/report";
import {
  cacheHfPreset,
  clearCachedHfPreset,
  derivePresetFromHfConfig,
  loadCachedHfPresets,
  loadModelConfig,
  loadSafetensorsTotal,
  loadStoredHfToken,
  searchModels,
  storeHfToken,
  type HfSearchHit,
} from "./lib/huggingface";
import { formatBytes, formatCompact, formatNumber, formatTime } from "./lib/units";
import type { HardwareCategory, HardwarePreset, ModelPreset, PrecisionMode, ScenarioInputs, ServingPlan } from "./types";
import "./styles.css";

type AppTab = "planner" | "docs" | "settings";

const precisionModes: Record<PrecisionMode, { label: string; bytes: number | null; note: string }> = {
  bf16: {
    label: "BF16 / FP16",
    bytes: 2,
    note: "2 bytes/param. Training default and the safest serving choice — full quality, every runtime supports it. Use when accuracy matters more than memory or you can't measure quality loss.",
  },
  fp8: {
    label: "FP8",
    bytes: 1,
    note: "1 byte/param. Production-ready on Hopper (H100/H200) and Blackwell. Halves weight memory vs BF16 with a small, well-characterized quality drop. Default for new vLLM deployments.",
  },
  fp4: {
    label: "FP4 / INT4",
    bytes: 0.5,
    note: "0.5 bytes/param. Quarters the weight footprint vs BF16. Native on Blackwell tensor cores; emulated elsewhere. Quality varies — validate with your eval set before committing.",
  },
  custom: {
    label: "Custom",
    bytes: null,
    note: "For mixed/quantized layouts (AWQ, GPTQ, marlin). Use the custom bytes-per-parameter field below.",
  },
};

const kvSliderMin = 512;
const kvSliderMax = 2_000_000;
const defaultServingBatch = 128;

// llama.cpp / LM Studio K and V cache quantization types. Bytes-per-element
// figures are derived from the GGUF block sizes (32 elements per block):
//   Q8_0 → 32 vals + 2-byte scale = 34/32 ≈ 1.0625
//   Q5_0 → 20 + 2 = 22/32 ≈ 0.6875        Q5_1 → 22 + 2 = 24/32 = 0.75
//   Q4_0 → 16 + 2 = 18/32 = 0.5625        Q4_1 → 16 + 2 + 2 = 20/32 = 0.625
//   IQ4_NL → same payload as Q4_0
// The two are picked independently in LM Studio, so we track them separately
// and let the calculator's kvBytesPerToken track the effective sum.
type KvQuantType = "f32" | "f16" | "q8_0" | "q5_1" | "q5_0" | "q4_1" | "q4_0" | "iq4_nl";

const kvQuantBytes: Record<KvQuantType, number> = {
  f32: 4.0,
  f16: 2.0,
  q8_0: 1.0625,
  q5_1: 0.75,
  q5_0: 0.6875,
  q4_1: 0.625,
  q4_0: 0.5625,
  iq4_nl: 0.5625,
};

const kvQuantLabels: Record<KvQuantType, string> = {
  f32: "F32",
  f16: "F16 (default)",
  q8_0: "Q8_0",
  q5_1: "Q5_1",
  q5_0: "Q5_0",
  q4_1: "Q4_1",
  q4_0: "Q4_0",
  iq4_nl: "IQ4_NL",
};

const kvQuantOrder: KvQuantType[] = ["f32", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"];

// A precision mode is hardware-supported when its bytes/param meet or exceed
// the hardware's smallest native tensor-core precision. Picking a narrower
// mode would only pretend to unlock more compute — see the clamp in
// getBatchThreshold. "Custom" is always allowed (escape hatch).
function isPrecisionSupported(mode: PrecisionMode, hardware: HardwarePreset): boolean {
  if (mode === "custom") return true;
  const bytes = precisionModes[mode].bytes;
  return bytes !== null && bytes >= hardware.nativeComputeBytes;
}

// Pick the lowest-byte precision the hardware natively supports — used as the
// fallback when the user switches to hardware that doesn't support the
// currently-selected precision.
function defaultPrecisionForHardware(hardware: HardwarePreset): PrecisionMode {
  const ordered: PrecisionMode[] = ["fp4", "fp8", "bf16"];
  return ordered.find((mode) => isPrecisionSupported(mode, hardware)) ?? "bf16";
}

// Display order + label for hardware categories. Used to group both the
// Planner Hardware <select> (via <optgroup>) and the Settings inventory.
const hardwareCategoryOrder: HardwareCategory[] = [
  "nvidia-blackwell",
  "nvidia-hopper",
  "nvidia-ampere",
  "nvidia-consumer",
  "nvidia-legacy",
  "apple-silicon",
];

const hardwareCategoryLabel: Record<HardwareCategory, string> = {
  "nvidia-blackwell": "NVIDIA Blackwell (B-series)",
  "nvidia-hopper": "NVIDIA Hopper (H-series)",
  "nvidia-ampere": "NVIDIA Ampere (A-series)",
  "nvidia-consumer": "NVIDIA RTX / consumer & workstation",
  "nvidia-legacy": "NVIDIA legacy (Volta)",
  "apple-silicon": "Apple Silicon (MacBook Pro / Mac mini)",
};

function groupHardwareByCategory(presets: HardwarePreset[]): Array<{ category: HardwareCategory; items: HardwarePreset[] }> {
  const buckets = new Map<HardwareCategory, HardwarePreset[]>();
  for (const preset of presets) {
    const list = buckets.get(preset.category) ?? [];
    list.push(preset);
    buckets.set(preset.category, list);
  }
  return hardwareCategoryOrder
    .filter((category) => buckets.has(category))
    .map((category) => ({ category, items: buckets.get(category)! }));
}

function App() {
  const defaultHardwareIds = ["dell-b300-8gpu", "h200-pool-16gpu", "h200-server-4gpu"];
  const [activeTab, setActiveTab] = useState<AppTab>("planner");
  const [hardwareId, setHardwareId] = useState("dell-b300-8gpu");
  const [enabledHardwareIds, setEnabledHardwareIds] = useState<string[]>(defaultHardwareIds);
  const [modelId, setModelId] = useState("qwen3-coder-next");
  const [precision, setPrecision] = useState<PrecisionMode>("fp4");
  const [contextTokens, setContextTokens] = useState(modelPresets[0].contextTokens);
  const [batchSize, setBatchSize] = useState(defaultServingBatch);
  const [customWeightBytes, setCustomWeightBytes] = useState(0.5);
  const [kvBytesPerToken, setKvBytesPerToken] = useState(modelPresets[0].kvBytesPerToken);
  // K and V cache quantization (matches LM Studio / llama.cpp). Both default
  // to F16 — that's the baseline every model preset's kvBytesPerToken assumes.
  const [kCacheType, setKCacheType] = useState<KvQuantType>("f16");
  const [vCacheType, setVCacheType] = useState<KvQuantType>("f16");
  const [tokensPerSecond, setTokensPerSecond] = useState(0);
  const [deploymentDays, setDeploymentDays] = useState(60);
  const [hfPresets, setHfPresets] = useState<ModelPreset[]>(() => loadCachedHfPresets());
  const [optimizeNotes, setOptimizeNotes] = useState<string[] | null>(null);
  const allModelPresets = useMemo<ModelPreset[]>(() => [...modelPresets, ...hfPresets], [hfPresets]);
  const [pipelineStages, setPipelineStages] = useState(1);
  const [expertParallelism, setExpertParallelism] = useState(8);
  const [safetyMargin, setSafetyMargin] = useState(0.8);
  // Lifecycle (optional). All zero by default → the lifecycle panel stays
  // hidden until the operator opts in by entering pretrain or RL tokens.
  const [pretrainTokens, setPretrainTokens] = useState(0);
  const [rlTokens, setRlTokens] = useState(0);
  const [rlInefficiency, setRlInefficiency] = useState(3);
  const [inferenceInefficiency, setInferenceInefficiency] = useState(5);

  const visibleHardwarePresets = hardwarePresets.filter((item) => enabledHardwareIds.includes(item.id));
  const plannerHardwarePresets = visibleHardwarePresets.length > 0 ? visibleHardwarePresets : hardwarePresets;
  const hardware = plannerHardwarePresets.find((item) => item.id === hardwareId) ?? plannerHardwarePresets[0];
  const model = allModelPresets.find((item) => item.id === modelId) ?? allModelPresets[0];
  const weightBytesPerParam = precisionModes[precision].bytes ?? customWeightBytes;

  const scenario = useMemo(
    () =>
      buildScenario(hardware, model, {
        contextTokens,
        batchSize,
        weightBytesPerParam,
        kvBytesPerToken,
        tokensPerSecond,
        deploymentDays,
        pipelineStages,
        expertParallelism,
        safetyMargin,
        pretrainTokens,
        rlTokens,
        rlInefficiency,
        inferenceInefficiency,
      }),
    [
      hardware,
      model,
      contextTokens,
      batchSize,
      weightBytesPerParam,
      kvBytesPerToken,
      tokensPerSecond,
      deploymentDays,
      pipelineStages,
      expertParallelism,
      safetyMargin,
      pretrainTokens,
      rlTokens,
      rlInefficiency,
      inferenceInefficiency,
    ],
  );
  const result = useMemo(() => calculateScenario(scenario), [scenario]);
  const comparison = useMemo(
    () =>
      plannerHardwarePresets.slice(0, 4).map((item) => {
        const compared = buildScenario(item, model, {
          contextTokens,
          batchSize,
          weightBytesPerParam,
          kvBytesPerToken,
          tokensPerSecond,
          deploymentDays,
          pipelineStages,
          expertParallelism: Math.min(expertParallelism, item.gpuCount),
          safetyMargin,
          pretrainTokens,
          rlTokens,
          rlInefficiency,
          inferenceInefficiency,
        });
        return { hardware: item, scenario: compared, result: calculateScenario(compared) };
      }),
    [
      model,
      plannerHardwarePresets,
      contextTokens,
      batchSize,
      weightBytesPerParam,
      kvBytesPerToken,
      tokensPerSecond,
      deploymentDays,
      pipelineStages,
      expertParallelism,
      safetyMargin,
      pretrainTokens,
      rlTokens,
      rlInefficiency,
      inferenceInefficiency,
    ],
  );
  const servingPlan = useMemo(() => createServingPlan(scenario, result), [scenario, result]);

  function applyHardware(nextId: string) {
    const next = hardwarePresets.find((item) => item.id === nextId) ?? hardwarePresets[0];
    setHardwareId(next.id);
    setExpertParallelism(next.gpuCount);
    // If the currently-selected precision isn't natively supported by the
    // new hardware, fall back to the smallest one that is. Custom is left
    // alone — the user is explicitly overriding storage precision there.
    setPrecision((current) => (isPrecisionSupported(current, next) ? current : defaultPrecisionForHardware(next)));
  }

  function toggleHardware(nextId: string) {
    setEnabledHardwareIds((current) => {
      const next = current.includes(nextId) ? current.filter((id) => id !== nextId) : [...current, nextId];
      if (next.length > 0 && !next.includes(hardwareId)) {
        applyHardware(next[0]);
      }
      return next;
    });
  }

  function setHardwareEnabled(ids: string[], enabled: boolean) {
    setEnabledHardwareIds((current) => {
      const set = new Set(current);
      for (const id of ids) {
        if (enabled) set.add(id);
        else set.delete(id);
      }
      const next = Array.from(set);
      if (next.length > 0 && !next.includes(hardwareId)) {
        applyHardware(next[0]);
      }
      return next;
    });
  }

  function applyModel(nextId: string) {
    const next = allModelPresets.find((item) => item.id === nextId) ?? allModelPresets[0];
    setModelId(next.id);
    setContextTokens(next.contextTokens);
    setKvBytesPerToken(next.kvBytesPerToken);
    // Each model preset's kvBytesPerToken is its F16/F16 baseline. Reset the
    // K/V quant pickers so subsequent quant changes scale from the baseline.
    setKCacheType("f16");
    setVCacheType("f16");
    setBatchSize(defaultServingBatch);
    setPrecision(next.defaultWeightBytesPerParam <= 0.5 ? "fp4" : next.defaultWeightBytesPerParam <= 1 ? "fp8" : "bf16");
    setCustomWeightBytes(next.defaultWeightBytesPerParam);
    setOptimizeNotes(null);
  }

  function applyAutoOptimize() {
    const out = autoOptimize(hardware, model, {
      contextTokens,
      weightBytesPerParam,
      kvBytesPerToken,
      batchSize,
      expertParallelism,
      pipelineStages,
      safetyMargin,
    });
    // Hardware, model, context, and weight precision are user-fixed — auto-optimize
    // only changes batch, KV cache, EP, PP, and safety margin.
    setKvBytesPerToken(out.overrides.kvBytesPerToken);
    setBatchSize(out.overrides.batchSize);
    setExpertParallelism(out.overrides.expertParallelism);
    setPipelineStages(out.overrides.pipelineStages);
    setSafetyMargin(out.overrides.safetyMargin);
    setOptimizeNotes(out.rationale);
  }

  function buildReport(): string {
    return buildScenarioReport({ hardware, model, scenario, result, plan: servingPlan });
  }

  function copyReport() {
    void navigator.clipboard?.writeText(buildReport());
  }

  function downloadReport() {
    const md = buildReport();
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const slug = `${hardware.id}_${model.id}`.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const date = new Date().toISOString().slice(0, 10);
    a.href = url;
    a.download = `dwarkoptimus-${slug}-${date}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <main className="app-shell">
      <div className="top-nav-row">
        <nav className="top-nav" aria-label="Primary">
          {(["planner", "docs", "settings"] as AppTab[]).map((tab) => (
            <button key={tab} type="button" className={activeTab === tab ? "active" : ""} onClick={() => setActiveTab(tab)}>
              {tab === "planner" ? "Planner" : tab === "docs" ? "Docs" : "Settings"}
            </button>
          ))}
        </nav>
        <a
          className="github-link"
          href="https://github.com/edicorAi/dwarkoptimus"
          target="_blank"
          rel="noreferrer noopener"
          aria-label="View source on GitHub"
          title="View source on GitHub"
        >
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <path
              fillRule="evenodd"
              clipRule="evenodd"
              fill="currentColor"
              d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56v-2.04c-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.27-1.69-1.27-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.69 1.24 3.34.95.1-.74.4-1.24.72-1.53-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.46.11-3.05 0 0 .96-.31 3.15 1.18.91-.25 1.89-.38 2.87-.39.97.01 1.95.14 2.86.39 2.18-1.49 3.14-1.18 3.14-1.18.62 1.59.23 2.76.11 3.05.74.81 1.18 1.84 1.18 3.1 0 4.43-2.7 5.4-5.27 5.68.41.36.78 1.07.78 2.16v3.21c0 .31.21.68.8.56C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z"
            />
          </svg>
          <span>GitHub</span>
        </a>
      </div>

      <header className="hero">
        <div>
          <p className="eyebrow">dwarkoptimus</p>
          <h1>Roofline math for serving LLMs on real GPUs</h1>
          <p className="dek">
            Pick a model and the hardware you actually have. Get a fit verdict, the real bottleneck
            (compute, weight bandwidth, or KV cache), the break-even batch size, and a vLLM command
            you can paste — backed by the same roofline math from Reiner Pope's blackboard lecture.
          </p>
        </div>
        <div className="hero-card">
          <span>Right now</span>
          <strong>{hardware.label}</strong>
          <small>{model.label}</small>
        </div>
      </header>

      {activeTab === "planner" && (
        <section className="layout">
          <aside className="input-panel">
            <div className="panel-heading planner-heading">
              <h2 className="section-title">Plan a deployment</h2>
              <button type="button" className="primary-button" onClick={applyAutoOptimize}>
                ⚡ Auto-optimize
              </button>
            </div>
            <FieldHint>
              Pick your hardware, model, context, and weight precision — those stay fixed. Auto-optimize then tunes batch, KV cache (with fp8 quantization if needed), parallelism, and safety margin to fit and minimize per-token cost.
            </FieldHint>
            {optimizeNotes && optimizeNotes.length > 0 && (
              <div className="optimize-notes">
                <strong>Auto-optimize applied</strong>
                <ul>
                  {optimizeNotes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
                <button type="button" className="link-button" onClick={() => setOptimizeNotes(null)}>
                  Dismiss
                </button>
              </div>
            )}
            <Field label="Hardware" help="Only hardware enabled in Settings appears here. Use Settings to match the planner to your inventory.">
              <select value={hardware.id} onChange={(event) => applyHardware(event.target.value)}>
                {groupHardwareByCategory(plannerHardwarePresets).map(({ category, items }) => (
                  <optgroup key={category} label={hardwareCategoryLabel[category]}>
                    {items.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </Field>
            <Field label="Model" help="Curated presets cover popular open-weights models. Frontier estimates are public guesses for closed-source models (GPT-5, Claude, Gemini, Grok) — useful as scale references, but you can't actually deploy them. Use the Hugging Face search to import any other model — KV cache size and context length are read directly from its config.json.">
              <select value={modelId} onChange={(event) => applyModel(event.target.value)}>
                <optgroup label="Open-weights presets">
                  {modelPresets.filter((item) => !item.id.startsWith("frontier-")).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </optgroup>
                <optgroup label="Frontier estimates (closed-source, not deployable)">
                  {modelPresets.filter((item) => item.id.startsWith("frontier-")).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </optgroup>
                {hfPresets.length > 0 && (
                  <optgroup label="Imported from Hugging Face">
                    {hfPresets.map((item) => (
                      <option key={item.id} value={item.id}>
                        🤗 {item.label}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            </Field>
            <HuggingFaceSearch
              onImport={(preset) => {
                cacheHfPreset(preset);
                setHfPresets((current) => {
                  const without = current.filter((item) => item.id !== preset.id);
                  return [preset, ...without];
                });
                setModelId(preset.id);
                setContextTokens(preset.contextTokens);
                setKvBytesPerToken(preset.kvBytesPerToken);
                setKCacheType("f16");
                setVCacheType("f16");
                setBatchSize(defaultServingBatch);
                setPrecision(
                  preset.defaultWeightBytesPerParam <= 0.5
                    ? "fp4"
                    : preset.defaultWeightBytesPerParam <= 1
                      ? "fp8"
                      : "bf16",
                );
                setCustomWeightBytes(preset.defaultWeightBytesPerParam);
              }}
              onForget={(presetId) => {
                clearCachedHfPreset(presetId);
                setHfPresets((current) => current.filter((item) => item.id !== presetId));
                if (modelId === presetId) {
                  setModelId(modelPresets[0].id);
                }
              }}
              importedIds={hfPresets.map((p) => p.id)}
            />
            <NumberField
              label="Context tokens"
              value={contextTokens}
              min={512}
              max={Math.max(model.contextTokens * 2, 8192)}
              step={512}
              onChange={setContextTokens}
              help="Prompt plus history length. Larger values increase KV cache pressure."
            />
            <NumberField
              label="Batch (concurrent sequences)"
              value={batchSize}
              min={1}
              max={Math.max(result.maxFittingBatch * 2, result.batchThreshold, 1024)}
              step={1}
              onChange={setBatchSize}
              help="How many users are decoded together in one step. This is the only concurrency knob — vLLM's --max-num-seqs comes from here."
            />

            <SectionTitle title="More controls" />

            <div className="field">
              <label>Weight precision</label>
              <FieldHint>
                Bytes used to store each model parameter. Smaller precision shrinks the weight footprint linearly (FP4 is ¼ the size of BF16) and unlocks more tensor-core throughput on hardware that supports it — but quality and kernel maturity get worse as you go down. Pick the smallest precision your model card and runtime actually support.
              </FieldHint>
              <div className="segmented">
                {(Object.keys(precisionModes) as PrecisionMode[]).map((mode) => {
                  const supported = isPrecisionSupported(mode, hardware);
                  const className = `${mode === precision ? "active" : ""} ${supported ? "" : "unsupported"}`.trim();
                  const title = supported
                    ? precisionModes[mode].label
                    : `${precisionModes[mode].label} — not natively supported on ${hardware.label}. Use Custom if you want to override.`;
                  return (
                    <button
                      key={mode}
                      type="button"
                      className={className}
                      onClick={() => setPrecision(mode)}
                      disabled={!supported}
                      title={title}
                    >
                      {precisionModes[mode].label}
                    </button>
                  );
                })}
              </div>
              <small>{precisionModes[precision].note}</small>
              <PrecisionExplainer
                weightBytesPerParam={weightBytesPerParam}
                hardware={hardware}
                model={model}
              />
            </div>
            {precision === "custom" && (
              <NumberField label="Custom bytes / param" value={customWeightBytes} min={0.1} max={4} step={0.1} onChange={setCustomWeightBytes} help="Manual storage precision for weights." />
            )}
            <KvQuantSelector
              kType={kCacheType}
              vType={vCacheType}
              onChange={(nextK, nextV) => {
                const oldRatio = kvQuantBytes[kCacheType] + kvQuantBytes[vCacheType];
                const newRatio = kvQuantBytes[nextK] + kvQuantBytes[nextV];
                setKCacheType(nextK);
                setVCacheType(nextV);
                if (oldRatio > 0) {
                  setKvBytesPerToken(Math.max(1, Math.round(kvBytesPerToken * (newRatio / oldRatio))));
                }
              }}
            />
            <KvBytesField value={kvBytesPerToken} onChange={setKvBytesPerToken} />
            <NumberField
              label="Pipeline stages"
              value={pipelineStages}
              min={1}
              max={Math.max(1, hardware.gpuCount)}
              step={1}
              onChange={setPipelineStages}
              help="Splits the model across racks. Saves weight memory per rack but does not reduce step time, and KV pressure does not shrink because micro-batches grow with depth. Usually left at 1 inside one NVLink domain."
            />
            <NumberField
              label="Expert parallelism"
              value={expertParallelism}
              min={1}
              max={hardware.gpuCount}
              step={1}
              onChange={setExpertParallelism}
              help="How many GPUs shard experts (MoE) or weights (dense) within a stage."
            />
            <NumberField
              label="Safety margin"
              value={safetyMargin}
              min={0.5}
              max={1}
              step={0.05}
              onChange={setSafetyMargin}
              help="Fraction of HBM you plan against. Reused as vLLM's --gpu-memory-utilization."
            />
            <NumberField
              label="Deployment days"
              value={deploymentDays}
              min={1}
              max={365}
              step={1}
              onChange={setDeploymentDays}
              help="Model serving lifetime. Only affects the Chinchilla coverage tile."
            />
            <NumberField
              label="Tokens / second (override)"
              value={tokensPerSecond}
              min={0}
              max={1_000_000}
              step={100}
              onChange={setTokensPerSecond}
              help="Leave at 0 to use the derived pool throughput (batch ÷ step). Override only if you have a measured rate."
            />

            <SectionTitle title="Lifecycle FLOPs (optional)" />
            <NumberField
              label="Pretrain tokens"
              value={pretrainTokens}
              min={0}
              max={1e15}
              step={1e11}
              onChange={setPretrainTokens}
              help="Tokens consumed during pretraining. Leave at 0 to hide the lifecycle panel. Pretrain FLOPs ≈ 6 × active_params × tokens."
            />
            <NumberField
              label="RL tokens"
              value={rlTokens}
              min={0}
              max={1e15}
              step={1e10}
              onChange={setRlTokens}
              help="Tokens consumed during RL post-training (rejection sampling + PPO/GRPO + reward model passes). RL FLOPs ≈ 2 × active_params × tokens × inefficiency."
            />
            <NumberField
              label="RL inefficiency"
              value={rlInefficiency}
              min={1}
              max={10}
              step={0.5}
              onChange={setRlInefficiency}
              help="Multiplier on the 2N forward baseline. ≈3 matches the 6N pretrain coefficient (forward + backward + RL overhead)."
            />
            <NumberField
              label="Decode inefficiency"
              value={inferenceInefficiency}
              min={1}
              max={20}
              step={0.5}
              onChange={setInferenceInefficiency}
              help="Multiplier capturing 'decode MFU is 1/N of prefill'. Default 5 matches the lecture's rule of thumb. Affects inference FLOPs in the lifecycle panel only."
            />
          </aside>

          <section className="content">
            <VerdictCard result={result} scenario={scenario} />
            <PlanningPanel
              plan={servingPlan}
              onApply={() => {
                const target = Math.max(1, Math.floor(servingPlan.maxFittingBatch));
                if (Number.isFinite(target) && target > 0) setBatchSize(target);
              }}
            />
            <MetricGrid result={result} scenario={scenario} />
            <LifecyclePanel result={result} />
            <div className="chart-grid">
              <MemoryChart result={result} hardware={hardware} />
              <LatencyChart scenario={scenario} batchThreshold={result.batchThreshold} />
              <CostChart scenario={scenario} batchThreshold={result.batchThreshold} />
            </div>
            <ComparisonTable rows={comparison} selectedHardwareId={hardware.id} />
            <Assumptions
              hardware={hardware}
              model={model}
              result={result}
              onCopyReport={copyReport}
              onDownloadReport={downloadReport}
            />
          </section>
        </section>
      )}

      {activeTab === "docs" && <DocsPanel />}

      {activeTab === "settings" && (
        <SettingsPanel
          enabledHardwareIds={enabledHardwareIds}
          selectedHardwareId={hardware.id}
          onToggleHardware={toggleHardware}
          onSelectHardware={applyHardware}
          onSetEnabled={setHardwareEnabled}
        />
      )}
    </main>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <h2 className="section-title">{title}</h2>;
}

function DocsPanel() {
  return (
    <section className="docs-shell">
      <article className="panel docs-intro">
        <p className="eyebrow">Documentation</p>
        <h2>How dwarkoptimus models LLM serving</h2>
        <p>
          Everything in this calculator comes from the roofline math in Reiner Pope's blackboard
          lecture on Dwarkesh Podcast (
          <a href="https://www.dwarkesh.com/p/reiner-pope" target="_blank" rel="noreferrer noopener">
            transcript
          </a>
          ,{" "}
          <a href="https://youtu.be/xmkSf5IS-zw" target="_blank" rel="noreferrer noopener">
            video
          </a>
          ). This page is a single-scroll reference for every concept, knob, and metric the app
          surfaces — so you can use the Planner without context-switching to figure out what a number
          means.
        </p>
      </article>

      <DocsToc />

      <div className="docs-content">
      <DocSection id="overview" title="What this app does">
        <p>
          You pick the GPU hardware you actually have and the LLM you want to serve. dwarkoptimus
          tells you four things, in plain language:
        </p>
        <ul>
          <li>
            <strong>Will it fit?</strong> Per-GPU memory pressure with a safety margin, with a
            green / yellow / red verdict.
          </li>
          <li>
            <strong>Where's the bottleneck?</strong> Compute, weight bandwidth, or KV cache
            memory.
          </li>
          <li>
            <strong>How big should the batch be?</strong> The break-even batch from roofline math
            and the largest batch HBM allows.
          </li>
          <li>
            <strong>What flags should I pass to vLLM?</strong> A copy-pasteable{" "}
            <code>vllm serve</code> command with the parallelism, memory, and KV settings derived
            from the verdict.
          </li>
        </ul>
        <p>
          The math is the same on every page; the Planner is just the UI that surfaces it. The pure
          calculation engine lives in <code>src/lib/calculations.ts</code> and is unit-tested
          directly.
        </p>
      </DocSection>

      <DocSection id="roofline" title="The roofline model (one-liner physics)">
        <p>
          One decode step on a GPU pool takes whichever is larger of two times: how long compute
          takes, and how long memory traffic takes.
        </p>
        <pre className="docs-code">{`t_compute = 2 × batch × active_params / FLOPs_peak
t_memory  = (weight_bytes + batch × context × kv_bytes_per_token) / bandwidth
step_time = max(t_compute, t_memory)`}</pre>
        <p>
          Two consequences fall out immediately:
        </p>
        <ul>
          <li>
            <strong>Latency floor.</strong> The weight-fetch term is constant in batch (you load
            the model once per step regardless of how many users are batched together), so step
            time has a floor. That's the dashed grey line in the latency chart.
          </li>
          <li>
            <strong>Cost floor.</strong> Per-token cost is{" "}
            <code>step_time / batch</code>. As batch grows, the weight-fetch term divides out and
            cost flattens at the compute-bound floor. That's the cost-per-token chart's tail.
          </li>
        </ul>
      </DocSection>

      <DocSection id="break-even" title="Break-even batch (the most-cited formula)">
        <p>
          The smallest batch where compute time meets memory time. Below it, you're paying full
          weight-fetch cost per token; above it, your tensor cores are saturated and per-token
          cost doesn't keep dropping.
        </p>
        <pre className="docs-code">{`break_even_batch = flops_per_byte
                 × (total_params / active_params)
                 × (native_compute_bytes / weight_bytes_per_param)`}</pre>
        <ul>
          <li>
            <code>flops_per_byte</code> is a hardware constant (peak FLOPs at native precision /
            HBM bandwidth) — about <strong>300 on Hopper bf16</strong>,{" "}
            <strong>590 on H100 fp8</strong>, <strong>1875 on Blackwell fp4</strong>.
          </li>
          <li>
            <code>total_params / active_params</code> is the model's <strong>sparsity ratio</strong>
            : 1 for dense models, ~30 for DeepSeek-style large MoEs.
          </li>
          <li>
            The third term scales the threshold by precision: serving fp8 weights on fp4 hardware
            cuts the threshold in half (less compute available per byte read).
          </li>
        </ul>
        <p>
          Reiner's lecture: "<em>This actually gives you a ballpark which is remarkably accurate
          to practice.</em>" For DeepSeek V3 (671B/37B at bf16) on H800, this works out to ~5.4K.
          For Qwen3-Coder-Next on B300 fp4, it's ~50K.
        </p>
      </DocSection>

      <DocSection id="step-interval" title="Step interval and the 'train departs every X ms' framing">
        <p>
          The natural cadence of decode steps equals the time to read all of HBM once:
        </p>
        <pre className="docs-code">step_interval ≈ HBM_capacity / bandwidth</pre>
        <p>
          For a B300 (288 GB / 8 TB/s) it's ~36 ms; for an H200 (141 GB / 4.8 TB/s) it's ~29 ms.
          Faster is physically impossible (you can't read all weights in less time than bandwidth
          allows). Slower would mean the GPU is sitting on its FLOPs idle.
        </p>
        <p>
          From this falls a useful derived quantity:
        </p>
        <pre className="docs-code">pool_throughput ≈ batch / step_interval</pre>
        <p>
          That's why the Planner lets you leave Tokens / second at 0 — it derives the realistic
          rate for your batch and hardware, then uses it for Chinchilla coverage.
        </p>
      </DocSection>

      <DocSection id="kv-cache" title="KV cache: usually the limiting factor">
        <p>
          During autoregressive decode, every previously generated token contributes a per-layer
          K and V vector that must be re-read on every step. Per token, that's:
        </p>
        <pre className="docs-code">kv_bytes_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes</pre>
        <p>
          For a Llama-3-8B with 32 layers × 8 KV heads × 128 head_dim × 2 (bf16) = <strong>128
          KB/token</strong>. Multiply by context length and batch and KV cache typically dwarfs
          weights at long context. That's why the auto-optimize button reaches for{" "}
          <code>--kv-cache-dtype fp8</code> as a fit-of-last-resort: halving KV bytes/token is the
          single highest-leverage memory move in vLLM.
        </p>
        <p>
          Some architectures cut this dramatically:
        </p>
        <ul>
          <li>
            <strong>MLA</strong> (DeepSeek V3, Mistral Large 3, Kimi K2): replace K/V with a small
            shared latent (~512 dim) plus a positional rope head. KV ends up around 70 KB/token at
            fp8 even for a 1T-param model.
          </li>
          <li>
            <strong>Sliding-window attention</strong> (Gemma): only the last N tokens hold full
            KV — older tokens drop out. Effective KV is bounded.
          </li>
          <li>
            <strong>Hybrid Mamba/transformer</strong> (Nemotron-3-Super, MiniMax): only
            attention layers contribute KV; Mamba layers carry a small fixed state. Roughly halves
            real KV.
          </li>
        </ul>
      </DocSection>

      <DocSection id="memory-fit" title="Memory fit (per GPU, with safety margin)">
        <p>
          The model's total memory footprint is sharded across the GPU pool. Fit is decided{" "}
          <em>per GPU</em> because each card is its own HBM domain — a 2 TB total demand fails on
          8× 288 GB GPUs even though pool capacity is 2.3 TB, if 250 GB lands on one card.
        </p>
        <pre className="docs-code">{`required_per_gpu = (weight_bytes + kv_bytes) / (EP × PP)
available_per_gpu = HBM × safety_margin
utilization = required_per_gpu / available_per_gpu`}</pre>
        <p>
          The "Memory fit" panel surfaces both per-GPU and pool-wide totals because conflating
          them is the most common confusion.
        </p>
      </DocSection>

      <DocSection id="bottleneck" title="Bottleneck classifier">
        <p>For decoder workloads the verdict tile reports one of three:</p>
        <ul>
          <li>
            <strong>kv-memory</strong> — KV cache exceeds 1.4× weight memory. Long context with
            full attention. Try shorter context, KV quantization, or an MLA model.
          </li>
          <li>
            <strong>weight-memory</strong> — total memory traffic dominates the compute side of
            the roofline. You're below the break-even batch, so the GPU spends most of its time
            streaming weights. Bump batch, drop precision, or accept that you're below traffic
            volume to amortize.
          </li>
          <li>
            <strong>compute</strong> — past the break-even batch, you're compute-bound and
            per-token cost stops dropping. Healthy operating point.
          </li>
        </ul>
        <p>
          For embedding and VLM presets the verdict is <strong>not-applicable</strong>: KV-cache
          decode math doesn't describe their workload.
        </p>
      </DocSection>

      <DocSection id="metrics" title="Every metric tile, explained">
        <dl className="docs-dl">
          <dt>Memory used / GPU</dt>
          <dd>
            <code>required_per_gpu</code>; the percentage is of the safety-adjusted available
            HBM.
          </dd>

          <dt>Max users at this context</dt>
          <dd>
            Largest batch HBM allows at the chosen context. Different from the break-even batch:
            this is a capacity ceiling, not an efficiency target.
          </dd>

          <dt>Break-even batch</dt>
          <dd>
            Cost-optimal batch from the roofline. Going above doesn't make per-token cost drop
            further; going below makes it climb.
          </dd>

          <dt>Step interval</dt>
          <dd>
            Time per decode step ≈ <code>HBM_capacity / bandwidth</code>. Lower bound on latency
            per generated token.
          </dd>

          <dt>Pool throughput (derived)</dt>
          <dd>
            <code>batch / step_interval</code>. The realistic peak token-rate for the whole pool
            at the current batch.
          </dd>

          <dt>Sparsity</dt>
          <dd>
            <code>total_params / active_params</code>. 1× for dense models. Sets how high the
            break-even batch must go: a 30× sparse MoE wants 30× the batch to amortize its
            weight reads.
          </dd>

          <dt>HBM drain time</dt>
          <dd>
            Time to read every weight in HBM once. Equals step interval; named separately because
            it's the more familiar physics number.
          </dd>

          <dt>Chinchilla coverage</dt>
          <dd>
            <code>inference_tokens / (20 × active_params)</code>. 1× means you'll serve as many
            tokens as a Chinchilla-optimal training run for this active-param size. Modern
            frontier deployments hit 100× and up — the over-training case Reiner discusses in the
            RL section of the lecture.
          </dd>
        </dl>
      </DocSection>

      <DocSection id="inputs" title="Every input, explained">
        <dl className="docs-dl">
          <dt>Hardware</dt>
          <dd>
            One GPU pool from the curated list (filterable in Settings). Each preset carries{" "}
            <code>flops_per_byte</code> at the hardware's native precision, plus a{" "}
            <code>nativeComputeBytes</code> field used to scale the break-even formula when you
            pick a different weight precision.
          </dd>

          <dt>Model</dt>
          <dd>
            Either a curated preset or a Hugging Face import. HF imports read{" "}
            <code>config.json</code> directly and compute KV bytes/token exactly from{" "}
            <code>num_hidden_layers × num_key_value_heads × head_dim × dtype_bytes</code>.
          </dd>

          <dt>Context tokens</dt>
          <dd>
            Prompt + history length. Larger values make the KV cache term grow linearly per
            sequence.
          </dd>

          <dt>Batch (concurrent sequences)</dt>
          <dd>
            How many users are decoded together each step. The only concurrency knob — the vLLM
            command's <code>--max-num-seqs</code> comes from here.
          </dd>

          <dt>Weight precision</dt>
          <dd>
            Bytes per parameter when storing weights. FP4 ≈ 0.5, FP8 ≈ 1, BF16 ≈ 2. Smaller
            shrinks the weight footprint linearly and unlocks more tensor-core throughput on
            hardware that supports it natively. Outside native precision, kernels dequantize on
            the fly.
          </dd>

          <dt>KV bytes / token</dt>
          <dd>
            Effective per-token KV cache bytes after any quantization or attention-architecture
            tricks. Use the slider when you know something the model card doesn't (e.g. you'll
            run with <code>--kv-cache-dtype fp8</code>, or it's an MLA / sliding-window model).
          </dd>

          <dt>Pipeline stages</dt>
          <dd>
            Splits the model across racks. Saves weight memory per rack but does <em>not</em>{" "}
            reduce step time, and KV pressure doesn't shrink because micro-batches grow with
            depth. Usually 1 inside a single NVLink domain.
          </dd>

          <dt>Expert parallelism</dt>
          <dd>
            How many GPUs shard MoE experts (or, for dense models, weights generally) within a
            stage. Default = full GPU count.
          </dd>

          <dt>Safety margin</dt>
          <dd>
            Fraction of HBM you'll plan against. Reused as vLLM's{" "}
            <code>--gpu-memory-utilization</code> flag in the suggested command.
          </dd>

          <dt>Tokens / second (override)</dt>
          <dd>
            Leave at 0 to use the derived pool throughput. Override only if you have a measured
            serving rate from production traces. Used for Chinchilla coverage only.
          </dd>

          <dt>Deployment days</dt>
          <dd>
            Model serving lifetime. Only affects Chinchilla coverage.
          </dd>
        </dl>
      </DocSection>

      <DocSection id="auto-optimize" title="Auto-optimize: what it does and doesn't touch">
        <p>
          One green button at the top of the Planner. Treats your hardware, model, context, and
          weight precision as user-fixed; tunes everything else.
        </p>
        <ul>
          <li>
            Sets <strong>expert parallelism</strong> = full GPU count (use the whole NVLink
            domain).
          </li>
          <li>
            Resets <strong>pipeline stages</strong> to 1 (PP doesn't help latency).
          </li>
          <li>
            Sets <strong>safety margin</strong> to 0.85 (~15% HBM headroom).
          </li>
          <li>
            <strong>KV quantization fallback:</strong> only when the workload doesn't fit and the
            weights themselves do fit, halve KV bytes/token (models{" "}
            <code>--kv-cache-dtype fp8</code>). If still no fit, quarter (4-bit KV).
          </li>
          <li>
            Picks <strong>batch</strong> = <code>min(break_even, max_fitting)</code>. Cost-optimal
            when HBM has room; clamped to fit otherwise.
          </li>
        </ul>
        <p>
          When even maxed-KV-quantization can't fit, Auto-optimize doesn't silently reduce context
          or precision — it tells you those are inputs <em>you</em> control. Stay in charge of
          decisions that affect quality.
        </p>
      </DocSection>

      <DocSection id="vllm-command" title="The suggested vLLM command">
        <p>
          The <em>Suggested vLLM config</em> panel emits a copy-pasteable{" "}
          <code>vllm serve</code> command derived from your scenario. Each flag maps to one
          calculator field:
        </p>
        <ul>
          <li>
            <code>--tensor-parallel-size</code> ← <code>gpu_count / pipeline_stages</code>
          </li>
          <li>
            <code>--pipeline-parallel-size</code> ← only emitted when PP &gt; 1
          </li>
          <li>
            <code>--max-model-len</code> ← context tokens
          </li>
          <li>
            <code>--gpu-memory-utilization</code> ← safety margin
          </li>
          <li>
            <code>--max-num-seqs</code> ← batch (concurrent sequences)
          </li>
          <li>
            <code>--max-num-batched-tokens</code> ← capped at 1M to avoid aggressive long-prefill
            defaults
          </li>
          <li>
            <code>--kv-cache-dtype fp8</code> ← only when KV bytes/token is below ~75% of the
            model's preset value (Auto-optimize triggers this)
          </li>
          <li>
            <code>--enable-prefix-caching</code> ← always on (cached input tokens are ~10× cheaper)
          </li>
        </ul>
      </DocSection>

      <DocSection id="hf-import" title="Hugging Face import">
        <p>
          The search box under the Model dropdown queries{" "}
          <code>huggingface.co/api/models</code>. Each result shows downloads, likes, and a 🔒
          gated badge if the repo requires license acceptance. Clicking <strong>Import</strong>:
        </p>
        <ol>
          <li>
            Fetches <code>config.json</code> from{" "}
            <code>huggingface.co/&lt;repo&gt;/resolve/main/config.json</code>.
          </li>
          <li>
            Reads <code>safetensors.total</code> (when published) for an authoritative param
            count, otherwise estimates from <code>hidden_size × layers × intermediate_size</code>.
          </li>
          <li>
            Computes KV bytes/token <em>exactly</em> from{" "}
            <code>2 × layers × kv_heads × head_dim × dtype_bytes</code> — no estimation.
          </li>
          <li>
            Saves the derived preset to <code>localStorage</code> (key prefix{" "}
            <code>dwarkoptimus.hf-preset.</code>), so it survives reloads.
          </li>
        </ol>
        <p>
          Gated repos (Llama, Gemma) need an HF access token: paste one into the field that
          appears on import failure. The token is stored in localStorage and reused for search +
          config fetches.
        </p>
      </DocSection>

      <DocSection id="report" title="Markdown report">
        <p>
          The <strong>Download report</strong> button in Assumptions exports a self-contained
          markdown file: verdict, hardware, model, configuration, memory fit (per-GPU + pool),
          roofline metrics, the vLLM command, all warnings, and a methodology section with the
          formulas. Filename:{" "}
          <code>dwarkoptimus-{"<hardware>_<model>"}-YYYY-MM-DD.md</code>.
        </p>
        <p>
          Use <strong>Copy report</strong> to put the same content on the clipboard for
          pasting into a doc.
        </p>
      </DocSection>

      <DocSection id="confidence" title="Confidence badges">
        <p>Each model and hardware preset carries one of four confidence levels:</p>
        <dl className="docs-dl">
          <dt>source-backed</dt>
          <dd>Direct from a vendor spec sheet or model card.</dd>
          <dt>estimated</dt>
          <dd>Derived from architecture details (e.g. KV bytes computed from layers × heads).</dd>
          <dt>user-provided</dt>
          <dd>Local hardware inventory or operator override.</dd>
          <dt>unknown</dt>
          <dd>Placeholder pending validation.</dd>
        </dl>
        <p>
          The Assumptions panel shows them as colored pills; the markdown report carries them too.
          Always check before turning a verdict into a deployment decision.
        </p>
      </DocSection>

      <DocSection id="glossary" title="Glossary">
        <dl className="docs-dl">
          <dt>Active params</dt>
          <dd>Parameters touched by a single token. Equals total params for dense models; smaller for MoE.</dd>
          <dt>BF16 / FP16 / FP8 / FP4</dt>
          <dd>Floating-point precisions: 2, 2, 1, 0.5 bytes per parameter respectively.</dd>
          <dt>Chinchilla optimum</dt>
          <dd>~20 training tokens per parameter (Hoffmann et al. 2022). The "coverage" tile compares serving lifetime to that benchmark.</dd>
          <dt>Decode</dt>
          <dd>The autoregressive step where the model produces one new token by reading the full KV history.</dd>
          <dt>FLOPs/byte</dt>
          <dd>Hardware ratio: peak FLOPs at native precision divided by memory bandwidth. Sets the break-even batch.</dd>
          <dt>GQA — Grouped-Query Attention</dt>
          <dd>KV heads &lt; query heads, sharing K and V across groups. Cuts KV cache by the head ratio.</dd>
          <dt>HBM</dt>
          <dd>High-Bandwidth Memory — the on-package GPU memory whose capacity and bandwidth drive most of this calculator.</dd>
          <dt>MLA — Multi-head Latent Attention</dt>
          <dd>DeepSeek's KV trick: replace K/V with a tiny shared latent + positional rope head. Slashes KV cache by ~10×.</dd>
          <dt>MoE — Mixture of Experts</dt>
          <dd>Each token activates a small subset of FFN expert sub-networks. Increases total params at fixed compute.</dd>
          <dt>NVLink / scale-up</dt>
          <dd>Fast intra-rack interconnect (~8× faster than scale-out network). All-to-all MoE traffic strongly prefers staying inside one NVLink domain.</dd>
          <dt>Pipeline parallelism (PP)</dt>
          <dd>Different layers on different GPUs, processed sequentially. Saves capacity, doesn't reduce step time.</dd>
          <dt>Prefill</dt>
          <dd>The pass over the input prompt before decode begins. Compute-bound and amenable to large effective batches.</dd>
          <dt>Sparsity ratio</dt>
          <dd>total_params / active_params. The factor by which the break-even batch grows.</dd>
          <dt>Tensor parallelism (TP)</dt>
          <dd>Sharding individual matmuls across GPUs in a single layer. The default vLLM parallel dimension.</dd>
        </dl>
      </DocSection>

      <DocSection id="sources" title="Sources">
        <ul>
          <li>
            <a href="https://www.dwarkesh.com/p/reiner-pope" target="_blank" rel="noreferrer noopener">
              Reiner Pope on Dwarkesh Podcast — full transcript
            </a>
          </li>
          <li>
            <a href="https://youtu.be/xmkSf5IS-zw" target="_blank" rel="noreferrer noopener">
              Same lecture on YouTube
            </a>
          </li>
          <li>
            <a href="https://en.wikipedia.org/wiki/Roofline_model" target="_blank" rel="noreferrer noopener">
              Roofline model — Wikipedia
            </a>
          </li>
          <li>
            <a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noreferrer noopener">
              Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla)
            </a>
          </li>
          <li>
            <a href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noreferrer noopener">
              DeepSeek-V3 technical report (MLA architecture)
            </a>
          </li>
          <li>
            <a href="https://docs.vllm.ai/" target="_blank" rel="noreferrer noopener">
              vLLM documentation — flag reference
            </a>
          </li>
        </ul>
      </DocSection>
      </div>
    </section>
  );
}

function DocsToc() {
  const sections: Array<[string, string]> = [
    ["overview", "What this app does"],
    ["roofline", "Roofline model"],
    ["break-even", "Break-even batch"],
    ["step-interval", "Step interval"],
    ["kv-cache", "KV cache"],
    ["memory-fit", "Memory fit"],
    ["bottleneck", "Bottleneck classifier"],
    ["metrics", "Metric reference"],
    ["inputs", "Input reference"],
    ["auto-optimize", "Auto-optimize"],
    ["vllm-command", "vLLM command"],
    ["hf-import", "Hugging Face import"],
    ["report", "Markdown report"],
    ["confidence", "Confidence badges"],
    ["glossary", "Glossary"],
    ["sources", "Sources"],
  ];
  return (
    <nav className="panel docs-toc" aria-label="Documentation contents">
      <h3>On this page</h3>
      <ul>
        {sections.map(([id, label]) => (
          <li key={id}>
            <a href={`#${id}`}>{label}</a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function DocSection({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <article className="panel doc-section" id={id}>
      <h2>
        <a href={`#${id}`} className="doc-anchor" aria-label={`Link to ${title}`}>
          #
        </a>{" "}
        {title}
      </h2>
      {children}
    </article>
  );
}

function SettingsPanel({
  enabledHardwareIds,
  selectedHardwareId,
  onToggleHardware,
  onSelectHardware,
  onSetEnabled,
}: {
  enabledHardwareIds: string[];
  selectedHardwareId: string;
  onToggleHardware: (id: string) => void;
  onSelectHardware: (id: string) => void;
  onSetEnabled: (ids: string[], enabled: boolean) => void;
}) {
  const selectedCount = enabledHardwareIds.length;
  const grouped = groupHardwareByCategory(hardwarePresets);

  return (
    <section className="settings-shell">
      <article className="panel settings-intro">
        <div>
          <h2>Settings</h2>
          <p>
            Choose the hardware you actually have. The Planner hardware dropdown and comparison table will only use
            the selected inventory. Each section below is a hardware family — toggle individual SKUs or enable/disable
            a whole family at once.
          </p>
        </div>
        <Badge tone={selectedCount > 0 ? "user-provided" : "unknown"}>{selectedCount} enabled</Badge>
      </article>

      {grouped.map(({ category, items }) => {
        const ids = items.map((item) => item.id);
        const enabledInGroup = ids.filter((id) => enabledHardwareIds.includes(id)).length;
        const allEnabled = enabledInGroup === ids.length;
        return (
          <article key={category} className="panel">
            <div className="panel-heading">
              <div>
                <h2>{hardwareCategoryLabel[category]}</h2>
                <span>
                  {enabledInGroup} of {ids.length} enabled
                </span>
              </div>
              <button
                type="button"
                className="secondary-button"
                onClick={() => onSetEnabled(ids, !allEnabled)}
              >
                {allEnabled ? "Disable all" : "Enable all"}
              </button>
            </div>
            <div className="hardware-catalog">
              {items.map((item) => {
                const enabled = enabledHardwareIds.includes(item.id);
                return (
                  <label key={item.id} className={`hardware-card ${enabled ? "enabled" : ""}`}>
                    <input type="checkbox" checked={enabled} onChange={() => onToggleHardware(item.id)} />
                    <div>
                      <strong>{item.label}</strong>
                      <span>
                        {item.gpuCount} GPU · {formatBytes(item.memoryBytesPerGpu)} / GPU · {formatBytes(item.memoryBandwidthBytesPerSecondPerGpu)}/s
                      </span>
                      <small>{item.notes}</small>
                    </div>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => onSelectHardware(item.id)}
                      disabled={!enabled || item.id === selectedHardwareId}
                    >
                      {item.id === selectedHardwareId ? "Active" : "Use"}
                    </button>
                  </label>
                );
              })}
            </div>
          </article>
        );
      })}
    </section>
  );
}

function Field({ label, help, children }: { label: string; help: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      <FieldHint>{help}</FieldHint>
      {children}
    </div>
  );
}

function NumberField({
  label,
  help,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  help: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  const safeMax = Math.max(min, max);
  const boundedValue = clampNumber(value, min, safeMax);
  const update = (nextValue: number) => onChange(clampNumber(nextValue, min, safeMax));

  return (
    <div className="field">
      <label>{label}</label>
      <FieldHint>{help}</FieldHint>
      <input type="number" value={boundedValue} min={min} max={safeMax} step={step} onChange={(event) => update(Number(event.target.value))} />
      <input type="range" value={boundedValue} min={min} max={safeMax} step={step} onChange={(event) => update(Number(event.target.value))} />
    </div>
  );
}

function KvBytesField({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const boundedValue = clampNumber(value, 0, kvSliderMax);
  const sliderValue = kvToSlider(boundedValue);
  const updateTypedValue = (nextValue: number) => onChange(clampNumber(nextValue, 0, kvSliderMax));

  return (
    <div className="field">
      <label>KV bytes / token</label>
      <FieldHint>
        Estimated cache per context token. The slider uses a log scale because useful KV values range from tiny MLA caches to multi-MB dense caches.
      </FieldHint>
      <input
        type="number"
        value={boundedValue}
        min={0}
        max={kvSliderMax}
        step={256}
        onChange={(event) => updateTypedValue(Number(event.target.value))}
      />
      <input
        type="range"
        value={sliderValue}
        min={0}
        max={100}
        step={1}
        onChange={(event) => onChange(sliderToKv(Number(event.target.value)))}
      />
      <small>Slider range: 0 to {formatBytes(kvSliderMax)} per token. Current: {formatBytes(boundedValue)}.</small>
    </div>
  );
}

function FieldHint({ children }: { children: React.ReactNode }) {
  return <p className="field-hint">{children}</p>;
}

function KvQuantSelector({
  kType,
  vType,
  onChange,
}: {
  kType: KvQuantType;
  vType: KvQuantType;
  onChange: (kType: KvQuantType, vType: KvQuantType) => void;
}) {
  const ratio = (kvQuantBytes[kType] + kvQuantBytes[vType]) / (2 * kvQuantBytes.f16);
  return (
    <div className="field">
      <label>KV cache quantization</label>
      <FieldHint>
        K and V cache types as exposed by llama.cpp / LM Studio. Picking
        narrower quants shrinks the per-token KV footprint — the slider
        below updates automatically. Some models lose quality at Q4 KV;
        treat this as a memory experiment, not a free win.
      </FieldHint>
      <div className="kv-quant-row">
        <label className="kv-quant-cell">
          <span>K cache</span>
          <select value={kType} onChange={(event) => onChange(event.target.value as KvQuantType, vType)}>
            {kvQuantOrder.map((type) => (
              <option key={type} value={type}>
                {kvQuantLabels[type]} ({kvQuantBytes[type]} B/elem)
              </option>
            ))}
          </select>
        </label>
        <label className="kv-quant-cell">
          <span>V cache</span>
          <select value={vType} onChange={(event) => onChange(kType, event.target.value as KvQuantType)}>
            {kvQuantOrder.map((type) => (
              <option key={type} value={type}>
                {kvQuantLabels[type]} ({kvQuantBytes[type]} B/elem)
              </option>
            ))}
          </select>
        </label>
      </div>
      <small>
        KV bytes/token multiplier: <strong>{ratio.toFixed(2)}×</strong> the F16/F16 baseline.
        {kType !== "f16" || vType !== "f16"
          ? " Requires Flash Attention in llama.cpp/LM Studio."
          : ""}
      </small>
    </div>
  );
}

function PrecisionExplainer({
  weightBytesPerParam,
  hardware,
  model,
}: {
  weightBytesPerParam: number;
  hardware: HardwarePreset;
  model: ModelPreset;
}) {
  const totalWeightBytes = model.totalParams * weightBytesPerParam;
  const perGpuWeightBytes = totalWeightBytes / Math.max(1, hardware.gpuCount);
  const nativeBytes = hardware.nativeComputeBytes;
  const nativeLabel = nativeBytes <= 0.5 ? "FP4" : nativeBytes <= 1 ? "FP8" : "BF16";
  const computeRatio = nativeBytes / weightBytesPerParam;
  let advisory: string | null = null;
  let tone: "ok" | "warn" | "info" = "ok";
  if (Math.abs(computeRatio - 1) < 0.01) {
    advisory = `Matches this hardware's native compute precision (${nativeLabel}) — full tensor-core throughput.`;
    tone = "ok";
  } else if (computeRatio < 1) {
    const lossFactor = 1 / computeRatio;
    advisory = `Wider than ${nativeLabel}: you're leaving roughly ${lossFactor.toFixed(0)}× tensor-core throughput on the table. Drop precision (or pick hardware whose native is BF16/FP8) for full speed.`;
    tone = "warn";
  } else {
    advisory = `Narrower than ${nativeLabel} native: kernels typically dequantize on the fly. Memory savings are real, but compute speed and quality depend on the runtime's quantization path (e.g. AWQ, GPTQ, NVFP4).`;
    tone = "info";
  }
  return (
    <div className={`precision-explainer ${tone}`}>
      <div className="precision-explainer-stats">
        <span>
          <strong>{formatBytes(totalWeightBytes)}</strong> weights total
        </span>
        <span>
          <strong>{formatBytes(perGpuWeightBytes)}</strong> / GPU
          {hardware.gpuCount > 1 ? ` (× ${hardware.gpuCount})` : ""}
        </span>
        <span>
          <strong>{weightBytesPerParam} B</strong> / param
        </span>
      </div>
      <p>{advisory}</p>
    </div>
  );
}

function HuggingFaceSearch({
  onImport,
  onForget,
  importedIds,
}: {
  onImport: (preset: ModelPreset) => void;
  onForget: (presetId: string) => void;
  importedIds: string[];
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HfSearchHit[]>([]);
  const [searching, setSearching] = useState(false);
  const [busyRepoId, setBusyRepoId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorRepoId, setErrorRepoId] = useState<string | null>(null);
  const [token, setToken] = useState<string>(() => loadStoredHfToken());
  const [showTokenField, setShowTokenField] = useState<boolean>(() => loadStoredHfToken().length > 0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setError(null);
    setErrorRepoId(null);
    if (!query.trim()) {
      setResults([]);
      setSearching(false);
      return;
    }
    setSearching(true);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const handle = window.setTimeout(async () => {
      try {
        const hits = await searchModels(query, { signal: controller.signal, token: token || undefined });
        if (!controller.signal.aborted) setResults(hits);
      } catch (err) {
        if (controller.signal.aborted) return;
        setResults([]);
        setError(err instanceof Error ? err.message : "Search failed");
      } finally {
        if (!controller.signal.aborted) setSearching(false);
      }
    }, 300);
    return () => {
      window.clearTimeout(handle);
      controller.abort();
    };
  }, [query, token]);

  function persistToken(next: string) {
    setToken(next);
    storeHfToken(next);
  }

  async function importRepo(repoId: string) {
    setBusyRepoId(repoId);
    setError(null);
    setErrorRepoId(null);
    try {
      const [config, knownTotal] = await Promise.all([
        loadModelConfig(repoId, { token: token || undefined }),
        loadSafetensorsTotal(repoId, { token: token || undefined }).catch(() => undefined),
      ]);
      const preset = derivePresetFromHfConfig(repoId, config, knownTotal);
      onImport(preset);
      setResults([]);
      setQuery("");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Import failed";
      setError(message);
      setErrorRepoId(repoId);
      // Auto-reveal token field when access is the problem.
      if (/gated|token/i.test(message)) setShowTokenField(true);
    } finally {
      setBusyRepoId(null);
    }
  }

  return (
    <div className="hf-search">
      <input
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search Hugging Face — e.g. qwen3, llama, deepseek"
        aria-label="Search Hugging Face for a model"
      />
      <div className="hf-token-row">
        <button
          type="button"
          className="link-button"
          onClick={() => setShowTokenField((current) => !current)}
        >
          {showTokenField ? "Hide token" : token ? "Token saved · edit" : "Add HF access token"}
        </button>
      </div>
      {showTokenField && (
        <div className="hf-token-field">
          <input
            type="password"
            value={token}
            onChange={(event) => persistToken(event.target.value)}
            placeholder="hf_…"
            aria-label="Hugging Face access token"
            autoComplete="off"
            spellCheck={false}
          />
          <small>
            Stored in your browser's localStorage. Generate one at{" "}
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noreferrer noopener"
            >
              huggingface.co/settings/tokens
            </a>
            . You still need to accept each gated model's license on its model page.
          </small>
        </div>
      )}
      {error && (
        <p className="hf-error">
          {errorRepoId ? <strong>{errorRepoId}: </strong> : null}
          {error}
        </p>
      )}
      {searching && <p className="hf-status">Searching huggingface.co…</p>}
      {results.length > 0 && (
        <ul className="hf-results">
          {results.map((hit) => {
            const id = hit.id ?? hit.modelId ?? "";
            const alreadyImported = importedIds.includes(`hf:${id}`);
            const busy = busyRepoId === id;
            const gated = Boolean(hit.gated) && hit.gated !== false;
            const gatedWithoutToken = gated && !token;
            return (
              <li key={id}>
                <div className="hf-result-meta">
                  <strong>
                    {gated && <span className="hf-tag">🔒 gated</span>}
                    {id}
                  </strong>
                  <small>
                    {typeof hit.downloads === "number" ? `${formatCompact(hit.downloads)} downloads` : ""}
                    {typeof hit.likes === "number" ? ` · ${formatCompact(hit.likes)} likes` : ""}
                    {gatedWithoutToken ? " · needs HF token" : ""}
                  </small>
                </div>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={busy}
                  title={
                    gatedWithoutToken
                      ? "This is a gated model. Add an HF token below and accept the license on its model page."
                      : undefined
                  }
                  onClick={() => (alreadyImported ? onForget(`hf:${id}`) : importRepo(id))}
                >
                  {busy ? "Loading…" : alreadyImported ? "Remove" : gatedWithoutToken ? "Import (needs token)" : "Import"}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function VerdictCard({ result, scenario }: { result: ReturnType<typeof calculateScenario>; scenario: ScenarioInputs }) {
  const lightLabel =
    result.verdict === "fits"
      ? "go"
      : result.verdict === "tight"
        ? "caution"
        : result.verdict === "does-not-fit"
          ? "stop"
          : "n/a";
  return (
    <article className={`verdict-card ${result.verdict}`}>
      <div>
        <p className="eyebrow">Operator verdict</p>
        <h2>
          <span
            className="verdict-light"
            role="img"
            aria-label={`${lightLabel}: ${result.verdictLabel}`}
            title={lightLabel}
          >
            <i className="red" />
            <i className="amber" />
            <i className="green" />
          </span>
          <span>{result.verdictLabel}</span>
        </h2>
        <p>{result.mainReason}</p>
      </div>
      <div className="next-action">
        <span>Next action</span>
        <strong>{result.nextAction}</strong>
      </div>
      <div className="verdict-facts">
        <Fact label="Model" value={scenario.model.label} />
        <Fact label="Hardware" value={scenario.hardware.label} />
        <Fact label="Bottleneck" value={result.bottleneck.replace("-", " ")} />
      </div>
    </article>
  );
}

function MetricGrid({ result, scenario }: { result: ReturnType<typeof calculateScenario>; scenario: ScenarioInputs }) {
  const stepLabel = Number.isFinite(result.stepIntervalSeconds)
    ? formatTime(result.stepIntervalSeconds)
    : "--";
  const tpsLabel = Number.isFinite(result.derivedTokensPerSecond)
    ? formatCompact(result.derivedTokensPerSecond, " tok/s")
    : "--";
  const sparsityLabel = Number.isFinite(result.sparsityRatio) && result.sparsityRatio > 0
    ? `${formatNumber(result.sparsityRatio, 1)}×`
    : "--";
  const chinchillaLabel = Number.isFinite(result.chinchillaRatio) && result.chinchillaRatio > 0
    ? `${formatNumber(result.chinchillaRatio, 2)}×`
    : "--";
  return (
    <section className="metric-grid">
      <Metric
        title="Memory used / GPU"
        value={formatBytes(result.requiredBytesPerGpu)}
        sub={`${formatNumber(result.memoryUtilization * 100, 0)}% of the safety-adjusted HBM budget`}
      />
      <Metric
        title="Max users at this context"
        value={formatCompact(result.maxFittingBatch)}
        sub="biggest batch the HBM budget allows"
      />
      <Metric
        title="Break-even batch"
        value={formatCompact(result.batchThreshold)}
        sub="below this, per-token cost climbs fast"
      />
      <Metric
        title="Step interval"
        value={stepLabel}
        sub="≈ HBM drain — the train departs this often"
      />
      <Metric
        title="Pool throughput"
        value={tpsLabel}
        sub="tokens / second derived from batch ÷ step"
      />
      <Metric
        title="Sparsity"
        value={sparsityLabel}
        sub="total / active params — the bigger this is, the more users you need to fill a batch"
      />
      <Metric
        title="HBM drain time"
        value={formatTime(result.hbmDrainSeconds)}
        sub="time to read every weight in HBM once"
      />
      <Metric
        title="Chinchilla coverage"
        value={chinchillaLabel}
        sub="lifetime served / 20·active params (1× = trained-equivalent)"
      />
      <Metric
        title="Prefill throughput"
        value={
          Number.isFinite(result.prefillTokensPerSecond)
            ? formatCompact(result.prefillTokensPerSecond, " tok/s")
            : "--"
        }
        sub="compute-bound asymptote — what prefill achieves at full FLOPs"
      />
      <Metric
        title="Decode MFU"
        value={
          Number.isFinite(result.decodeMfu)
            ? `${formatNumber(result.decodeMfu * 100, 1)}%`
            : "--"
        }
        sub="fraction of step time actually doing FLOPs (rest is HBM waiting)"
      />
      <Metric
        title="Crossover context"
        value={
          Number.isFinite(result.crossoverContextTokens)
            ? `${formatCompact(result.crossoverContextTokens)} tok`
            : "--"
        }
        sub="below: compute-bound. above: KV-bandwidth-bound"
      />
      {scenario.pipelineStages > 1 && (
        <Metric
          title="Pipeline efficiency"
          value={`${formatNumber(result.pipelineEfficiency * 100, 0)}%`}
          sub={`bubble ${formatNumber(result.pipelineBubbleFraction * 100, 0)}% — raise batch or drop PP to recover`}
        />
      )}
    </section>
  );
}

function LifecyclePanel({ result }: { result: ReturnType<typeof calculateScenario> }) {
  const lc = result.lifecycle;
  if (lc.totalFlops <= 0) return null;
  const phaseLabel: Record<typeof lc.dominantPhase, string> = {
    pretrain: "Pretrain",
    rl: "RL",
    inference: "Inference",
    none: "—",
  };
  const dominanceLabel = Number.isFinite(lc.dominanceRatio)
    ? `${formatNumber(lc.dominanceRatio, 2)}× the runner-up`
    : "no runner-up to compare";
  const bar = (share: number, label: string, sub: string, tone: string) => (
    <div className={`lifecycle-row ${tone}`}>
      <div className="lifecycle-row-head">
        <strong>{label}</strong>
        <span>{formatNumber(share * 100, 1)}%</span>
      </div>
      <div className="lifecycle-bar">
        <i style={{ width: `${Math.min(100, share * 100)}%` }} />
      </div>
      <small>{sub}</small>
    </div>
  );
  return (
    <article className="panel lifecycle-panel">
      <div className="panel-heading">
        <h2>Lifecycle FLOPs</h2>
        <span>
          Dominant: <strong>{phaseLabel[lc.dominantPhase]}</strong> ({dominanceLabel})
        </span>
      </div>
      <p className="lifecycle-note">
        Reiner's three-phase accounting. The compute-optimal equilibrium is roughly{" "}
        <code>D_pretrain ≈ 1.5·D_RL ≈ D_inference</code> in token-equivalents. A single phase
        running away means the deployment is over- or under-invested in that phase.
      </p>
      <div className="lifecycle-bars">
        {bar(
          lc.pretrainShare,
          "Pretrain",
          `${formatCompact(lc.pretrainFlops)} FLOPs (6·N·D)`,
          "pretrain",
        )}
        {bar(lc.rlShare, "RL", `${formatCompact(lc.rlFlops)} FLOPs (2·N·D × ineff.)`, "rl")}
        {bar(
          lc.inferenceShare,
          "Inference",
          `${formatCompact(lc.inferenceFlops)} FLOPs (2·N·D × ineff., decode-MFU penalty)`,
          "inference",
        )}
      </div>
    </article>
  );
}

function Metric({ title, value, sub }: { title: string; value: string; sub: string }) {
  return (
    <article className="metric">
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{sub}</small>
    </article>
  );
}

function PlanningPanel({ plan, onApply }: { plan: ServingPlan; onApply: () => void }) {
  function copyCommand() {
    void navigator.clipboard?.writeText(plan.command);
  }

  const target = Math.max(1, Math.floor(plan.maxFittingBatch));
  const noChange = !Number.isFinite(plan.maxFittingBatch) || target <= 0 || target === plan.requestedBatch;

  return (
    <article className={`panel planning-panel ${plan.fitsRequestedBatch ? "fits" : "does-not-fit"}`}>
      <div className="panel-heading">
        <div>
          <h2>Suggested vLLM config</h2>
          <span>{plan.summary}</span>
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={onApply}
          disabled={noChange}
          title={
            noChange
              ? "Batch is already at the largest size that fits"
              : `Set batch to ${target.toLocaleString()}`
          }
        >
          {noChange ? `At max (${target.toLocaleString()})` : `Use largest fitting batch (${target.toLocaleString()})`}
        </button>
      </div>

      <div className="planning-grid">
        <Fact label="Selected batch" value={formatCompact(plan.requestedBatch)} />
        <Fact label="Planned batch" value={formatCompact(plan.plannedBatch)} />
        <Fact label="Max that fits" value={formatCompact(plan.maxFittingBatch)} />
        <Fact label="TP size" value={String(plan.recommendedTensorParallelSize)} />
      </div>

      <div className="vllm-block">
        <div className="panel-heading">
          <h3>Suggested vLLM settings</h3>
          <button type="button" className="secondary-button" onClick={copyCommand}>
            Copy command
          </button>
        </div>
        <pre>{plan.command}</pre>
        <small>
          Heuristic defaults: `--max-num-seqs` follows planned concurrency, `--max-model-len` follows selected context,
          and `--max-num-batched-tokens` is capped at 1M to avoid an aggressive long-prefill default.
        </small>
      </div>

      {plan.warnings.length > 0 && (
        <ul>
          {plan.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </article>
  );
}

function ComparisonTable({
  rows,
  selectedHardwareId,
}: {
  rows: Array<{ hardware: HardwarePreset; scenario: ScenarioInputs; result: ReturnType<typeof calculateScenario> }>;
  selectedHardwareId: string;
}) {
  return (
    <article className="panel">
      <div className="panel-heading">
        <h2>Hardware comparison</h2>
        <span>same model and batch</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Hardware</th>
              <th>Verdict</th>
              <th>Used / GPU</th>
              <th>Remaining / GPU</th>
              <th>Drain</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ hardware, result }) => (
              <tr key={hardware.id} className={hardware.id === selectedHardwareId ? "selected-row" : ""}>
                <td>{hardware.label}</td>
                <td><Badge tone={result.verdict}>{result.verdictLabel.split(" on ")[0]}</Badge></td>
                <td>{formatBytes(result.requiredBytesPerGpu)}</td>
                <td>{formatBytes(result.memoryRemainingBytesPerGpu)}</td>
                <td>{formatTime(result.hbmDrainSeconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function MemoryChart({
  result,
  hardware,
}: {
  result: ReturnType<typeof calculateScenario>;
  hardware: HardwarePreset;
}) {
  const usedPct = clamp(result.requiredBytesPerGpu / result.availableBytesPerGpu, 0, 1.4) * 100;
  const remainingPct = Math.max(0, 100 - usedPct);
  const gpuCount = hardware.gpuCount;
  const poolMultiplier = gpuCount > 1 ? ` (× ${gpuCount} GPUs)` : "";
  return (
    <article className="panel">
      <div className="panel-heading">
        <h2>Memory fit</h2>
        <span>{formatBytes(result.availableBytesPerGpu)} usable / GPU{poolMultiplier}</span>
      </div>
      <div className="memory-bar" aria-label="Memory utilization">
        <i style={{ width: `${Math.min(usedPct, 100)}%` }} />
        <b style={{ width: `${remainingPct}%` }} />
      </div>
      <div className="memory-legend">
        <span><i className="used" /> Required {formatBytes(result.requiredBytesPerGpu)}</span>
        <span><i className="free" /> Remaining {formatBytes(result.memoryRemainingBytesPerGpu)}</span>
      </div>
      {gpuCount > 1 && (
        <div className="memory-pool-row">
          <table className="memory-pool-table">
            <thead>
              <tr>
                <th></th>
                <th>Per GPU</th>
                <th>Pool (× {gpuCount})</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Weights</td>
                <td>{formatBytes(result.weightBytesTotal / gpuCount)}</td>
                <td>{formatBytes(result.weightBytesTotal)}</td>
              </tr>
              <tr>
                <td>KV cache</td>
                <td>{formatBytes(result.kvBytesTotal / gpuCount)}</td>
                <td>{formatBytes(result.kvBytesTotal)}</td>
              </tr>
              <tr>
                <td>Required</td>
                <td>{formatBytes(result.requiredBytesPerGpu)}</td>
                <td>{formatBytes(result.requiredBytesPerGpu * gpuCount)}</td>
              </tr>
              <tr>
                <td>Available (safety-adj)</td>
                <td>{formatBytes(result.availableBytesPerGpu)}</td>
                <td>{formatBytes(result.availableBytesPerGpu * gpuCount)}</td>
              </tr>
            </tbody>
          </table>
          <p className="chart-caption">
            The model is sharded across all {gpuCount} GPUs — each one holds {formatBytes(result.weightBytesTotal / gpuCount)} of weights and {formatBytes(result.kvBytesTotal / gpuCount)} of KV. Fit is decided per GPU because each card is its own HBM domain.
          </p>
        </div>
      )}
    </article>
  );
}

// Plot rectangle in SVG units. Both charts share the same axes layout.
const PLOT = { left: 56, right: 510, top: 36, bottom: 200, width: 454, height: 164 };
const TICK_FRACTIONS = [0, 0.25, 0.5, 0.75, 1];

function AxisGrid({
  maxX,
  maxY,
  xFormat,
  yFormat,
  yUnit,
}: {
  maxX: number;
  maxY: number;
  xFormat: (v: number) => string;
  yFormat: (v: number) => string;
  yUnit?: string;
}) {
  return (
    <g>
      {TICK_FRACTIONS.map((f) => {
        const y = PLOT.bottom - f * PLOT.height;
        return (
          <g key={`y-${f}`}>
            <line x1={PLOT.left} y1={y} x2={PLOT.right} y2={y} className="grid-line" />
            <text x={PLOT.left - 6} y={y + 3} className="axis-tick" textAnchor="end">
              {yFormat(f * maxY)}
            </text>
          </g>
        );
      })}
      {TICK_FRACTIONS.map((f) => {
        const x = PLOT.left + f * PLOT.width;
        return (
          <g key={`x-${f}`}>
            <line x1={x} y1={PLOT.top} x2={x} y2={PLOT.bottom} className="grid-line" />
            <text x={x} y={PLOT.bottom + 14} className="axis-tick" textAnchor="middle">
              {xFormat(f * maxX)}
            </text>
          </g>
        );
      })}
      {/* solid axes on top of the grid */}
      <line x1={PLOT.left} y1={PLOT.top} x2={PLOT.left} y2={PLOT.bottom} className="axis-line" />
      <line x1={PLOT.left} y1={PLOT.bottom} x2={PLOT.right} y2={PLOT.bottom} className="axis-line" />
      <text x={PLOT.right} y={PLOT.bottom + 28} className="axis-label" textAnchor="end">
        batch
      </text>
      {yUnit && (
        <text x={PLOT.left - 6} y={PLOT.top - 10} className="axis-label" textAnchor="end">
          {yUnit}
        </text>
      )}
    </g>
  );
}

function LatencyChart({ scenario, batchThreshold }: { scenario: ScenarioInputs; batchThreshold: number }) {
  const points = getRooflineSweep(scenario);
  const maxX = Math.max(...points.map((p) => p.batch));
  const maxY = Math.max(...points.map((p) => p.totalSeconds), 1e-9);
  const weightFloor = points[0]?.weightFetchSeconds ?? 0;
  const path = (key: "computeSeconds" | "kvFetchSeconds" | "totalSeconds") =>
    points
      .map((point, index) => {
        const x = PLOT.left + (point.batch / maxX) * PLOT.width;
        const y = PLOT.bottom - (point[key] / maxY) * PLOT.height;
        return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  const weightFloorY = PLOT.bottom - (weightFloor / maxY) * PLOT.height;
  const knee =
    Number.isFinite(batchThreshold) && batchThreshold > 0 && batchThreshold < maxX
      ? PLOT.left + (batchThreshold / maxX) * PLOT.width
      : null;

  return (
    <article className="panel">
      <div className="panel-heading">
        <h2>Latency vs batch</h2>
        <span>One decode step as batch grows. Total = max(compute, weight-fetch + KV-fetch)</span>
      </div>
      <svg viewBox="0 0 560 240" className="latency-svg" role="img" aria-label="Latency vs batch chart">
        <AxisGrid maxX={maxX} maxY={maxY} xFormat={(v) => formatCompact(v)} yFormat={(v) => formatTime(v)} yUnit="step time" />
        {knee !== null && (
          <g>
            <line x1={knee} y1={PLOT.top} x2={knee} y2={PLOT.bottom} className="knee-line" />
            <text x={knee + 4} y={PLOT.top + 10} className="knee-label">
              break-even ≈ {formatCompact(batchThreshold)}
            </text>
          </g>
        )}
        {/* weight fetch is constant in batch — horizontal floor */}
        <line x1={PLOT.left} y1={weightFloorY} x2={PLOT.right} y2={weightFloorY} className="weight-line" />
        <text x={PLOT.left + 4} y={weightFloorY - 4} className="floor-label">
          weight fetch ≈ {formatTime(weightFloor)}
        </text>
        <path d={path("kvFetchSeconds")} className="memory-line" />
        <path d={path("computeSeconds")} className="compute-line" />
        <path d={path("totalSeconds")} className="total-line" />
      </svg>
      <div className="memory-legend">
        <span><i className="compute" /> t_compute</span>
        <span><i className="memory" /> KV fetch</span>
        <span><i className="weight" /> weight fetch</span>
        <span><i className="total" /> step time</span>
      </div>
      <p className="chart-caption">
        The weight-fetch line is flat — you load the full model regardless of batch — so it sets a floor on step time. KV fetch and compute both scale with batch; whichever is larger drives total step time.
      </p>
    </article>
  );
}

function CostChart({ scenario, batchThreshold }: { scenario: ScenarioInputs; batchThreshold: number }) {
  const points = getRooflineSweep(scenario);
  const maxX = Math.max(...points.map((p) => p.batch));
  const trimmed = points.filter((p) => Number.isFinite(p.costPerToken));
  // clamp the y range to the third sample onward — batch=1 explodes the hyperbola.
  const maxY = Math.max(...trimmed.slice(2).map((p) => p.costPerToken), 1e-9);
  const path = trimmed
    .map((point, index) => {
      const x = PLOT.left + (point.batch / maxX) * PLOT.width;
      const yRaw = Math.min(point.costPerToken, maxY);
      const y = PLOT.bottom - (yRaw / maxY) * PLOT.height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  const knee =
    Number.isFinite(batchThreshold) && batchThreshold > 0 && batchThreshold < maxX
      ? PLOT.left + (batchThreshold / maxX) * PLOT.width
      : null;
  const yFormat = (v: number) => `${formatTime(v)}/tok`;

  return (
    <article className="panel">
      <div className="panel-heading">
        <h2>Cost per token vs batch</h2>
        <span>Lower is cheaper. Flat tail = compute-bound</span>
      </div>
      <svg viewBox="0 0 560 240" className="latency-svg" role="img" aria-label="Cost per token vs batch chart">
        <AxisGrid maxX={maxX} maxY={maxY} xFormat={(v) => formatCompact(v)} yFormat={yFormat} yUnit="cost / tok" />
        {knee !== null && (
          <g>
            <line x1={knee} y1={PLOT.top} x2={knee} y2={PLOT.bottom} className="knee-line" />
            <text x={knee + 4} y={PLOT.top + 10} className="knee-label">
              amortization knee ≈ {formatCompact(batchThreshold)}
            </text>
          </g>
        )}
        <path d={path} className="total-line" />
      </svg>
      <p className="chart-caption">
        At small batches each user pays a full weight-fetch. The hyperbola flattens past the break-even batch — that flat line is the floor on per-token cost for this hardware.
      </p>
    </article>
  );
}

function Assumptions({
  hardware,
  model,
  result,
  onCopyReport,
  onDownloadReport,
}: {
  hardware: HardwarePreset;
  model: ModelPreset;
  result: ReturnType<typeof calculateScenario>;
  onCopyReport: () => void;
  onDownloadReport: () => void;
}) {
  return (
    <article className="panel assumptions">
      <div className="panel-heading">
        <h2>Assumptions and warnings</h2>
        <div className="report-actions">
          <button type="button" className="secondary-button" onClick={onCopyReport}>
            Copy report
          </button>
          <button type="button" className="primary-button" onClick={onDownloadReport}>
            ⬇ Download report
          </button>
        </div>
      </div>
      <div className="badge-row">
        <Badge tone={hardware.confidence}>Hardware: {hardware.confidence}</Badge>
        <Badge tone={model.confidence}>Model: {model.confidence}</Badge>
        <Badge tone={model.kvConfidence}>KV: {model.kvConfidence}</Badge>
      </div>
      <p>{hardware.notes}</p>
      <p>{model.notes}</p>
      {result.warnings.length > 0 && (
        <ul>
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </article>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Badge({ children, tone }: { children: React.ReactNode; tone: string }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

function kvToSlider(value: number): number {
  if (value <= 0) return 0;
  const boundedValue = clampNumber(value, kvSliderMin, kvSliderMax);
  const minLog = Math.log10(kvSliderMin);
  const maxLog = Math.log10(kvSliderMax);
  return ((Math.log10(boundedValue) - minLog) / (maxLog - minLog)) * 100;
}

function sliderToKv(value: number): number {
  if (value <= 0) return 0;
  const minLog = Math.log10(kvSliderMin);
  const maxLog = Math.log10(kvSliderMax);
  const raw = 10 ** (minLog + (value / 100) * (maxLog - minLog));
  return Math.round(raw / 256) * 256;
}

createRoot(document.getElementById("root")!).render(<App />);
