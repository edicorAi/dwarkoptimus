import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { hardwarePresets } from "./data/hardware";
import { modelPresets } from "./data/models";
import { buildScenario, calculateScenario, createServingPlan } from "./lib/calculations";
import { formatBytes, formatCompact, formatNumber, formatTime } from "./lib/units";
import type { HardwarePreset, ModelPreset, PrecisionMode, ScenarioInputs, ServingPlan } from "./types";
import "./styles.css";

type AppTab = "planner" | "advanced" | "settings";

const precisionModes: Record<PrecisionMode, { label: string; bytes: number | null; note: string }> = {
  bf16: { label: "BF16 / FP16", bytes: 2, note: "Highest compatibility, largest weight footprint." },
  fp8: { label: "FP8", bytes: 1, note: "Common serving precision for newer accelerators." },
  fp4: { label: "FP4 / INT4", bytes: 0.5, note: "Smallest preset footprint; quality and kernel support vary." },
  custom: { label: "Custom", bytes: null, note: "Use the custom bytes-per-parameter field." },
};

const kvSliderMin = 512;
const kvSliderMax = 2_000_000;
const defaultServingBatch = 128;

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
  const [tokensPerSecond, setTokensPerSecond] = useState(50e6);
  const [desiredConcurrentUsers, setDesiredConcurrentUsers] = useState(defaultServingBatch);
  const [deploymentDays, setDeploymentDays] = useState(60);
  const [pipelineStages, setPipelineStages] = useState(1);
  const [expertParallelism, setExpertParallelism] = useState(8);
  const [safetyMargin, setSafetyMargin] = useState(0.8);

  const visibleHardwarePresets = hardwarePresets.filter((item) => enabledHardwareIds.includes(item.id));
  const plannerHardwarePresets = visibleHardwarePresets.length > 0 ? visibleHardwarePresets : hardwarePresets;
  const hardware = plannerHardwarePresets.find((item) => item.id === hardwareId) ?? plannerHardwarePresets[0];
  const model = modelPresets.find((item) => item.id === modelId) ?? modelPresets[0];
  const weightBytesPerParam = precisionModes[precision].bytes ?? customWeightBytes;

  const scenario = useMemo(
    () =>
      buildScenario(hardware, model, {
        contextTokens,
        batchSize,
        weightBytesPerParam,
        kvBytesPerToken,
        tokensPerSecond,
        desiredConcurrentUsers,
        deploymentDays,
        pipelineStages,
        expertParallelism,
        safetyMargin,
      }),
    [
      hardware,
      model,
      contextTokens,
      batchSize,
      weightBytesPerParam,
      kvBytesPerToken,
      tokensPerSecond,
      desiredConcurrentUsers,
      deploymentDays,
      pipelineStages,
      expertParallelism,
      safetyMargin,
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
          desiredConcurrentUsers,
          deploymentDays,
          pipelineStages,
          expertParallelism: Math.min(expertParallelism, item.gpuCount),
          safetyMargin,
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
      desiredConcurrentUsers,
      deploymentDays,
      pipelineStages,
      expertParallelism,
      safetyMargin,
    ],
  );
  const servingPlan = useMemo(() => createServingPlan(scenario, result), [scenario, result]);

  function applyHardware(nextId: string) {
    const next = hardwarePresets.find((item) => item.id === nextId) ?? hardwarePresets[0];
    setHardwareId(next.id);
    setExpertParallelism(next.gpuCount);
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

  function applyModel(nextId: string) {
    const next = modelPresets.find((item) => item.id === nextId) ?? modelPresets[0];
    setModelId(next.id);
    setContextTokens(next.contextTokens);
    setKvBytesPerToken(next.kvBytesPerToken);
    setBatchSize(defaultServingBatch);
    setDesiredConcurrentUsers(defaultServingBatch);
    setPrecision(next.defaultWeightBytesPerParam <= 0.5 ? "fp4" : next.defaultWeightBytesPerParam <= 1 ? "fp8" : "bf16");
    setCustomWeightBytes(next.defaultWeightBytesPerParam);
  }

  function exportMarkdown() {
    const markdown = [
      `# dwarkoptimus Scenario`,
      ``,
      `- Hardware: ${hardware.label}`,
      `- Model: ${model.label}`,
      `- Verdict: ${result.verdictLabel}`,
      `- Reason: ${result.mainReason}`,
      `- Next action: ${result.nextAction}`,
      `- Context: ${formatCompact(contextTokens, " tokens")}`,
      `- Batch: ${formatCompact(batchSize)}`,
      `- Desired concurrent users: ${formatCompact(desiredConcurrentUsers)}`,
      `- Max fitting batch at selected context: ${formatCompact(result.maxFittingBatch)}`,
      `- Weight footprint: ${formatBytes(result.weightBytesTotal)}`,
      `- KV footprint: ${formatBytes(result.kvBytesTotal)}`,
      `- Required per GPU: ${formatBytes(result.requiredBytesPerGpu)}`,
      `- Safety-adjusted available per GPU: ${formatBytes(result.availableBytesPerGpu)}`,
      ``,
      `## Warnings`,
      ...result.warnings.map((warning) => `- ${warning}`),
    ].join("\n");
    void navigator.clipboard?.writeText(markdown);
  }

  return (
    <main className="app-shell">
      <nav className="top-nav" aria-label="Primary">
        {(["planner", "advanced", "settings"] as AppTab[]).map((tab) => (
          <button key={tab} type="button" className={activeTab === tab ? "active" : ""} onClick={() => setActiveTab(tab)}>
            {tab === "planner" ? "Planner" : tab === "advanced" ? "Advanced" : "Settings"}
          </button>
        ))}
      </nav>

      <header className="hero">
        <div>
          <p className="eyebrow">dwarkoptimus</p>
          <h1>Plan model serving on your GPUs</h1>
          <p className="dek">
            Pick a model and the NVIDIA hardware in your inventory. The app turns roofline math into a fit verdict,
            bottleneck explanation, and next action.
          </p>
        </div>
        <div className="hero-card">
          <span>Default view</span>
          <strong>{hardware.label}</strong>
          <small>{model.label}</small>
        </div>
      </header>

      {activeTab === "planner" && (
        <section className="layout">
          <aside className="input-panel">
            <SectionTitle title="Plan a deployment" />
            <Field label="Hardware" help="Only hardware enabled in Settings appears here. Use Settings to match the planner to your inventory.">
              <select value={hardware.id} onChange={(event) => applyHardware(event.target.value)}>
                {plannerHardwarePresets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Model" help="These names match the LiteLLM config. Some parameters are source-backed, while KV values are often estimates.">
              <select value={modelId} onChange={(event) => applyModel(event.target.value)}>
                {modelPresets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </Field>
            <NumberField label="Desired concurrent users" value={desiredConcurrentUsers} min={1} max={10000} step={1} onChange={setDesiredConcurrentUsers} help="How many live sequences/users you want this deployment to support at the selected context length." />
            <NumberField label="Context tokens" value={contextTokens} min={512} max={Math.max(model.contextTokens * 2, 8192)} step={512} onChange={setContextTokens} help="Prompt plus history length. Larger values increase KV cache pressure." />
            <NumberField label="Serving batch" value={batchSize} min={1} max={Math.max(result.maxFittingBatch * 2, result.batchThreshold, 1024)} step={1} onChange={setBatchSize} help="Concurrent sequences kept in flight for one decode step. This is a serving choice, not the theoretical weight-amortization threshold." />
          </aside>

          <section className="content">
            <VerdictCard result={result} scenario={scenario} />
            <PlanningPanel plan={servingPlan} onApply={() => setBatchSize(servingPlan.plannedConcurrency)} />
            <MetricGrid result={result} scenario={scenario} />
            <ComparisonTable rows={comparison} selectedHardwareId={hardware.id} />
          </section>
        </section>
      )}

      {activeTab === "advanced" && (
        <section className="layout">
          <aside className="input-panel">
            <SectionTitle title="Model memory" />
            <KvBytesField value={kvBytesPerToken} onChange={setKvBytesPerToken} />
            <div className="field">
              <label>Weight precision <Help text="Sets bytes per parameter for weight memory. FP4/INT4 is about 0.5 bytes, FP8 is 1 byte, BF16 is 2 bytes." /></label>
              <div className="segmented">
                {(Object.keys(precisionModes) as PrecisionMode[]).map((mode) => (
                  <button key={mode} type="button" className={mode === precision ? "active" : ""} onClick={() => setPrecision(mode)}>
                    {precisionModes[mode].label}
                  </button>
                ))}
              </div>
              <small>{precisionModes[precision].note}</small>
            </div>
            {precision === "custom" && (
              <NumberField label="Custom bytes / param" value={customWeightBytes} min={0.1} max={4} step={0.1} onChange={setCustomWeightBytes} help="Manual storage precision for weights." />
            )}

            <SectionTitle title="Inference and training" />
            <NumberField label="Tokens / second" value={tokensPerSecond} min={0} max={1e9} step={1e6} onChange={setTokensPerSecond} help="Expected serving rate for this model. Used for lifetime inference-token estimates." />
            <NumberField label="Deployment days" value={deploymentDays} min={1} max={365} step={1} onChange={setDeploymentDays} help="How long the model serves traffic before replacement." />
            <NumberField label="Pipeline stages" value={pipelineStages} min={1} max={Math.max(1, hardware.gpuCount)} step={1} onChange={setPipelineStages} help="Sequential model partitions. Helps weight capacity but does not magically remove KV pressure." />
            <NumberField label="Expert parallelism" value={expertParallelism} min={1} max={hardware.gpuCount} step={1} onChange={setExpertParallelism} help="How many GPUs shard experts or weights within a stage." />
            <NumberField label="Safety margin" value={safetyMargin} min={0.5} max={1} step={0.05} onChange={setSafetyMargin} help="Fraction of HBM you are willing to plan against. Lower values leave more runtime headroom." />
          </aside>

          <section className="content">
            <MetricGrid result={result} scenario={scenario} />
            <div className="chart-grid">
              <MemoryChart result={result} />
              <LatencyChart scenario={scenario} />
            </div>
            <Assumptions hardware={hardware} model={model} result={result} onExport={exportMarkdown} />
          </section>
        </section>
      )}

      {activeTab === "settings" && (
        <SettingsPanel
          enabledHardwareIds={enabledHardwareIds}
          selectedHardwareId={hardware.id}
          onToggleHardware={toggleHardware}
          onSelectHardware={applyHardware}
        />
      )}
    </main>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <h2 className="section-title">{title}</h2>;
}

function SettingsPanel({
  enabledHardwareIds,
  selectedHardwareId,
  onToggleHardware,
  onSelectHardware,
}: {
  enabledHardwareIds: string[];
  selectedHardwareId: string;
  onToggleHardware: (id: string) => void;
  onSelectHardware: (id: string) => void;
}) {
  const selectedCount = enabledHardwareIds.length;

  return (
    <section className="settings-shell">
      <article className="panel settings-intro">
        <div>
          <h2>Settings</h2>
          <p>
            Choose the NVIDIA hardware you actually have. The Planner hardware dropdown and comparison table will only use
            the selected inventory.
          </p>
        </div>
        <Badge tone={selectedCount > 0 ? "user-provided" : "unknown"}>{selectedCount} enabled</Badge>
      </article>

      <article className="panel">
        <div className="panel-heading">
          <div>
            <h2>Hardware inventory</h2>
            <span>NVIDIA accelerators and common server shapes</span>
          </div>
        </div>

        <div className="hardware-catalog">
          {hardwarePresets.map((item) => {
            const enabled = enabledHardwareIds.includes(item.id);
            return (
              <label key={item.id} className={`hardware-card ${enabled ? "enabled" : ""}`}>
                <input type="checkbox" checked={enabled} onChange={() => onToggleHardware(item.id)} />
                <div>
                  <strong>{item.label}</strong>
                  <span>
                    {item.gpuCount} GPU · {formatBytes(item.memoryBytesPerGpu)} HBM / GPU · {formatBytes(item.memoryBandwidthBytesPerSecondPerGpu)}/s
                  </span>
                  <small>{item.notes}</small>
                </div>
                <button type="button" className="secondary-button" onClick={() => onSelectHardware(item.id)} disabled={!enabled || item.id === selectedHardwareId}>
                  {item.id === selectedHardwareId ? "Active" : "Use"}
                </button>
              </label>
            );
          })}
        </div>
      </article>
    </section>
  );
}

function Field({ label, help, children }: { label: string; help: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>
        {label} <Help text={help} />
      </label>
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
      <label>
        {label} <Help text={help} />
      </label>
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
      <label>
        KV bytes / token{" "}
        <Help text="Estimated cache per context token. The slider uses a log scale because useful KV values range from tiny MLA caches to multi-MB dense caches." />
      </label>
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

function Help({ text }: { text: string }) {
  return (
    <span className="help" tabIndex={0} aria-label={text}>
      ?
      <span role="tooltip">{text}</span>
    </span>
  );
}

function VerdictCard({ result, scenario }: { result: ReturnType<typeof calculateScenario>; scenario: ScenarioInputs }) {
  return (
    <article className={`verdict-card ${result.verdict}`}>
      <div>
        <p className="eyebrow">Operator verdict</p>
        <h2>{result.verdictLabel}</h2>
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
  return (
    <section className="metric-grid">
      <Metric title="Memory used / GPU" value={formatBytes(result.requiredBytesPerGpu)} sub={`${formatNumber(result.memoryUtilization * 100, 0)}% of safety budget`} />
      <Metric title="Max fitting batch" value={formatCompact(result.maxFittingBatch)} sub="at selected context and KV size" />
      <Metric title="Batch threshold" value={formatCompact(result.batchThreshold)} sub="cost-optimal weight amortization target" />
      <Metric title="HBM drain time" value={formatTime(result.hbmDrainSeconds)} sub={`${scenario.hardware.gpuCount} GPU preset`} />
    </section>
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

  return (
    <article className={`panel planning-panel ${plan.fitsRequestedConcurrency ? "fits" : "does-not-fit"}`}>
      <div className="panel-heading">
        <div>
          <h2>Planning mode</h2>
          <span>{plan.summary}</span>
        </div>
        <button type="button" className="secondary-button" onClick={onApply}>
          Apply planned batch
        </button>
      </div>

      <div className="planning-grid">
        <Fact label="Requested users" value={formatCompact(plan.requestedConcurrency)} />
        <Fact label="Planned max users" value={formatCompact(plan.plannedConcurrency)} />
        <Fact label="Max that fits" value={formatCompact(plan.maxFittingConcurrency)} />
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

function MemoryChart({ result }: { result: ReturnType<typeof calculateScenario> }) {
  const usedPct = clamp(result.requiredBytesPerGpu / result.availableBytesPerGpu, 0, 1.4) * 100;
  const remainingPct = Math.max(0, 100 - usedPct);
  return (
    <article className="panel">
      <div className="panel-heading">
        <h2>Memory fit</h2>
        <span>{formatBytes(result.availableBytesPerGpu)} usable / GPU</span>
      </div>
      <div className="memory-bar" aria-label="Memory utilization">
        <i style={{ width: `${Math.min(usedPct, 100)}%` }} />
        <b style={{ width: `${remainingPct}%` }} />
      </div>
      <div className="memory-legend">
        <span><i className="used" /> Required {formatBytes(result.requiredBytesPerGpu)}</span>
        <span><i className="free" /> Remaining {formatBytes(result.memoryRemainingBytesPerGpu)}</span>
      </div>
    </article>
  );
}

function LatencyChart({ scenario }: { scenario: ScenarioInputs }) {
  const points = Array.from({ length: 32 }, (_, index) => {
    const batch = 1 + (index / 31) * Math.max(scenario.batchSize * 1.5, 100);
    const memorySeconds =
      (scenario.model.totalParams * scenario.weightBytesPerParam + batch * scenario.contextTokens * scenario.kvBytesPerToken) /
      (scenario.hardware.memoryBandwidthBytesPerSecondPerGpu * scenario.hardware.gpuCount);
    const computeSeconds =
      (batch * scenario.model.activeParams) /
      (scenario.flopsPerByte * scenario.hardware.memoryBandwidthBytesPerSecondPerGpu * scenario.hardware.gpuCount);
    return { batch, memorySeconds, computeSeconds, totalSeconds: Math.max(memorySeconds, computeSeconds) };
  });
  const maxX = Math.max(...points.map((p) => p.batch));
  const maxY = Math.max(...points.map((p) => p.totalSeconds), 1e-9);
  const path = (key: "memorySeconds" | "computeSeconds" | "totalSeconds") =>
    points
      .map((point, index) => {
        const x = 48 + (point.batch / maxX) * 452;
        const y = 190 - (point[key] / maxY) * 150;
        return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");

  return (
    <article className="panel">
      <div className="panel-heading">
        <h2>Latency shape</h2>
        <span>y-axis: {formatTime(maxY)}</span>
      </div>
      <svg viewBox="0 0 540 230" className="latency-svg" role="img" aria-label="Latency shape chart">
        <line x1="48" y1="40" x2="48" y2="190" />
        <line x1="48" y1="190" x2="500" y2="190" />
        <text x="12" y="48">{formatTime(maxY)}</text>
        <text x="450" y="216">batch</text>
        <path d={path("computeSeconds")} className="compute-line" />
        <path d={path("memorySeconds")} className="memory-line" />
        <path d={path("totalSeconds")} className="total-line" />
      </svg>
      <div className="memory-legend">
        <span><i className="compute" /> compute</span>
        <span><i className="memory" /> memory</span>
        <span><i className="total" /> total</span>
      </div>
    </article>
  );
}

function Assumptions({
  hardware,
  model,
  result,
  onExport,
}: {
  hardware: HardwarePreset;
  model: ModelPreset;
  result: ReturnType<typeof calculateScenario>;
  onExport: () => void;
}) {
  return (
    <article className="panel assumptions">
      <div className="panel-heading">
        <h2>Assumptions and warnings</h2>
        <button type="button" className="secondary-button" onClick={onExport}>Copy scenario Markdown</button>
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
