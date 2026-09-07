import type { HardwarePreset, ModelPreset, ScenarioInputs, ScenarioResult, ServingPlan } from "../types";
import { formatBytes, formatCompact, formatNumber, formatTime, formatUsd } from "./units";

export type ReportInput = {
  hardware: HardwarePreset;
  model: ModelPreset;
  scenario: ScenarioInputs;
  result: ScenarioResult;
  plan: ServingPlan;
  generatedAt?: Date;
};

const precisionLabel = (bytes: number): string => {
  if (bytes <= 0.5) return "FP4 / INT4";
  if (bytes <= 1) return "FP8";
  if (bytes <= 2) return "BF16 / FP16";
  return `${bytes} B/param`;
};

const nativePrecisionLabel = (bytes: number): string => {
  if (bytes <= 0.5) return "fp4";
  if (bytes <= 1) return "fp8";
  return "bf16";
};

const verdictBadge = (verdict: string): string => {
  if (verdict === "fits") return "🟢 Fits";
  if (verdict === "tight") return "🟡 Tight";
  if (verdict === "does-not-fit") return "🔴 Does not fit";
  return "⚪ Not applicable";
};

const num = (n: number, label = ""): string =>
  Number.isFinite(n) ? `${formatCompact(n)}${label}` : "—";

const timeOrDash = (n: number): string => (Number.isFinite(n) ? formatTime(n) : "—");

const bytesOrDash = (n: number): string => (Number.isFinite(n) ? formatBytes(n) : "—");

export function buildScenarioReport(input: ReportInput): string {
  const { hardware, model, scenario, result, plan } = input;
  const generatedAt = input.generatedAt ?? new Date();
  const lines: string[] = [];

  lines.push("# dwarkoptimus serving report");
  lines.push("");
  lines.push(`Generated ${generatedAt.toISOString()}`);
  lines.push("");

  lines.push("## Verdict");
  lines.push("");
  lines.push(`**${verdictBadge(result.verdict)} — ${result.verdictLabel}**`);
  lines.push("");
  lines.push(`- Reason: ${result.mainReason}`);
  lines.push(`- Next action: ${result.nextAction}`);
  lines.push(`- Bottleneck: ${result.bottleneck.replace("-", " ")}`);
  lines.push("");

  lines.push("## Hardware");
  lines.push("");
  lines.push(`- **${hardware.label}**`);
  lines.push(`- GPU count: ${hardware.gpuCount}`);
  lines.push(`- HBM per GPU: ${formatBytes(hardware.memoryBytesPerGpu)}`);
  lines.push(`- Memory bandwidth: ${formatBytes(hardware.memoryBandwidthBytesPerSecondPerGpu)}/s per GPU`);
  lines.push(
    `- FLOPs/byte (${nativePrecisionLabel(hardware.nativeComputeBytes)}): ${hardware.flopsPerByte}`,
  );
  lines.push(`- Interconnect: ${hardware.interconnect}`);
  lines.push(`- Confidence: ${hardware.confidence}`);
  if (hardware.notes) lines.push(`- Notes: ${hardware.notes}`);
  lines.push("");

  lines.push("## Model");
  lines.push("");
  lines.push(`- **${model.label}**`);
  lines.push(`- Architecture: ${model.architecture}`);
  lines.push(`- Total params: ${formatCompact(model.totalParams)}`);
  lines.push(
    `- Active params: ${formatCompact(model.activeParams)} (sparsity ${formatNumber(result.sparsityRatio, 2)}×)`,
  );
  if (model.activatedExperts) lines.push(`- Activated experts / token: ${model.activatedExperts}`);
  if (model.approxLayers) lines.push(`- Approx transformer layers: ${model.approxLayers}`);
  lines.push(`- Native context: ${formatCompact(model.contextTokens, " tokens")}`);
  lines.push(`- KV bytes / token (preset): ${formatBytes(model.kvBytesPerToken)}`);
  lines.push(`- Confidence: ${model.confidence} · KV confidence: ${model.kvConfidence}`);
  if (model.notes) lines.push(`- Notes: ${model.notes}`);
  lines.push("");

  lines.push("## Configuration");
  lines.push("");
  lines.push("| Knob | Value |");
  lines.push("|---|---|");
  lines.push(`| Context | ${formatCompact(scenario.contextTokens, " tokens")} |`);
  lines.push(`| Batch (concurrent sequences) | ${formatCompact(scenario.batchSize)} |`);
  lines.push(`| Weight precision | ${precisionLabel(scenario.weightBytesPerParam)} |`);
  lines.push(`| KV bytes / token (effective) | ${formatBytes(scenario.kvBytesPerToken)} |`);
  lines.push(`| Pipeline stages | ${scenario.pipelineStages} |`);
  lines.push(`| Expert parallelism | ${scenario.expertParallelism} |`);
  lines.push(`| Safety margin | ${scenario.safetyMargin} |`);
  if (scenario.costPerGpuHour > 0) {
    lines.push(`| GPU cost | ${formatUsd(scenario.costPerGpuHour)} / GPU·hour |`);
  }
  lines.push("");

  lines.push("## Memory fit");
  lines.push("");
  const gpuCount = hardware.gpuCount;
  const poolHeader = `Pool total (× ${gpuCount} GPU${gpuCount === 1 ? "" : "s"})`;
  const poolBytes = (perGpu: number) =>
    Number.isFinite(perGpu) ? bytesOrDash(perGpu * gpuCount) : "—";
  lines.push(`| Metric | Per GPU | ${poolHeader} |`);
  lines.push("|---|---|---|");
  lines.push(
    `| Weight footprint | ${bytesOrDash(result.weightBytesTotal / gpuCount)} | ${bytesOrDash(result.weightBytesTotal)} |`,
  );
  lines.push(
    `| KV footprint (at this batch × context) | ${bytesOrDash(result.kvBytesTotal / gpuCount)} | ${bytesOrDash(result.kvBytesTotal)} |`,
  );
  lines.push(
    `| Required | ${bytesOrDash(result.requiredBytesPerGpu)} | ${poolBytes(result.requiredBytesPerGpu)} |`,
  );
  lines.push(
    `| Safety-adjusted available | ${bytesOrDash(result.availableBytesPerGpu)} | ${poolBytes(result.availableBytesPerGpu)} |`,
  );
  lines.push(
    `| Remaining | ${bytesOrDash(result.memoryRemainingBytesPerGpu)} | ${poolBytes(result.memoryRemainingBytesPerGpu)} |`,
  );
  lines.push(
    `| Utilization | ${Number.isFinite(result.memoryUtilization) ? `${formatNumber(result.memoryUtilization * 100, 0)}%` : "—"} | same — fit is determined per GPU |`,
  );
  lines.push(`| Max fitting batch at this context (pool-wide) | — | ${num(result.maxFittingBatch)} |`);
  lines.push("");

  lines.push("## Roofline metrics");
  lines.push("");
  lines.push("| Metric | Value |");
  lines.push("|---|---|");
  lines.push(`| Break-even batch | ${num(result.batchThreshold)} |`);
  lines.push(`| HBM drain time | ${timeOrDash(result.hbmDrainSeconds)} |`);
  lines.push(`| Step interval (≈ HBM drain) | ${timeOrDash(result.stepIntervalSeconds)} |`);
  lines.push(`| Pool throughput (derived) | ${num(result.derivedTokensPerSecond, " tok/s")} |`);
  lines.push(
    `| Sparsity (total / active) | ${Number.isFinite(result.sparsityRatio) ? `${formatNumber(result.sparsityRatio, 2)}×` : "—"} |`,
  );
  lines.push(
    `| Chinchilla coverage | ${Number.isFinite(result.chinchillaRatio) ? `${formatNumber(result.chinchillaRatio, 2)}×` : "—"} |`,
  );
  lines.push(
    `| Prefill throughput (compute-bound) | ${num(result.prefillTokensPerSecond, " tok/s")} |`,
  );
  lines.push(
    `| Decode MFU (this batch) | ${Number.isFinite(result.decodeMfu) ? `${formatNumber(result.decodeMfu * 100, 1)}%` : "—"} |`,
  );
  lines.push(
    `| Crossover context (compute → memory bound) | ${Number.isFinite(result.crossoverContextTokens) ? `${formatCompact(result.crossoverContextTokens)} tok` : "—"} |`,
  );
  lines.push(`| Time to first token (compute-bound prefill) | ${timeOrDash(result.ttftSeconds)} |`);
  if (Number.isFinite(result.costPerMillionTokensUsd)) {
    lines.push(
      `| Serving cost (roofline step at this batch) | ${formatUsd(result.costPerMillionTokensUsd)} / 1M tokens at ${formatUsd(scenario.costPerGpuHour)}/GPU·hr × ${hardware.gpuCount} GPUs |`,
    );
  }
  if (scenario.pipelineStages > 1) {
    lines.push(
      `| Pipeline efficiency | ${formatNumber(result.pipelineEfficiency * 100, 1)}% (bubble ${formatNumber(result.pipelineBubbleFraction * 100, 1)}%) |`,
    );
  }
  lines.push("");

  if (result.lifecycle.totalFlops > 0) {
    lines.push("## Lifecycle FLOPs");
    lines.push("");
    lines.push("| Phase | FLOPs | Share |");
    lines.push("|---|---|---|");
    lines.push(
      `| Pretrain (6N·D) | ${formatCompact(result.lifecycle.pretrainFlops)} | ${formatNumber(result.lifecycle.pretrainShare * 100, 1)}% |`,
    );
    lines.push(
      `| RL (2N·D × ineff.) | ${formatCompact(result.lifecycle.rlFlops)} | ${formatNumber(result.lifecycle.rlShare * 100, 1)}% |`,
    );
    lines.push(
      `| Inference (2N·D × ineff.) | ${formatCompact(result.lifecycle.inferenceFlops)} | ${formatNumber(result.lifecycle.inferenceShare * 100, 1)}% |`,
    );
    lines.push("");
    lines.push(
      `Dominant phase: **${result.lifecycle.dominantPhase}** (${Number.isFinite(result.lifecycle.dominanceRatio) ? `${formatNumber(result.lifecycle.dominanceRatio, 2)}×` : "—"} the runner-up). Reiner's equilibrium is roughly D_pretrain ≈ 1.5 · D_RL ≈ D_inference.`,
    );
    lines.push("");
  }

  lines.push("## Suggested vLLM serve command");
  lines.push("");
  lines.push("```bash");
  lines.push(plan.command);
  lines.push("```");
  lines.push("");
  lines.push("Plan facts:");
  lines.push("");
  lines.push(`- Selected batch: ${formatCompact(plan.requestedBatch)}`);
  lines.push(`- Planned batch (after fit clamp): ${formatCompact(plan.plannedBatch)}`);
  lines.push(`- Max that fits: ${formatCompact(plan.maxFittingBatch)}`);
  lines.push(`- Tensor-parallel size: ${plan.recommendedTensorParallelSize}`);
  lines.push(
    `- gpu-memory-utilization: ${formatNumber(plan.recommendedGpuMemoryUtilization, 2)}`,
  );
  lines.push("");

  const allWarnings = [...result.warnings, ...plan.warnings];
  if (allWarnings.length > 0) {
    lines.push("## Warnings");
    lines.push("");
    for (const w of allWarnings) lines.push(`- ${w}`);
    lines.push("");
  }

  lines.push("## Methodology");
  lines.push("");
  lines.push(
    "Roofline math from Reiner Pope's blackboard lecture on Dwarkesh Podcast. The " +
      "calculator approximates one decode step as the larger of two times:",
  );
  lines.push("");
  lines.push("- `t_compute = 2 × batch × active_params / FLOPs_peak`");
  lines.push("- `t_memory = (weight_bytes + batch × context × kv_bytes_per_token) / bandwidth`");
  lines.push("- `step_time = max(t_compute, t_memory)`");
  lines.push(
    "- `break_even_batch = flops_per_byte × (total_params / active_params) × (native_compute_bytes / weight_bytes_per_param)`",
  );
  lines.push("- `step_interval ≈ HBM_capacity / bandwidth`  ← the train departs this often");
  lines.push("- `pool_throughput ≈ batch / step_interval`");
  lines.push("");

  return lines.join("\n");
}
