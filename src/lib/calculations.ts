import type { HardwarePreset, ModelPreset, ScenarioInputs, ScenarioResult, ServingPlan, Verdict } from "../types";

const decoderArchitectures = new Set(["dense", "moe", "hybrid"]);

export function buildScenario(
  hardware: HardwarePreset,
  model: ModelPreset,
  overrides: Partial<Omit<ScenarioInputs, "hardware" | "model">> = {},
): ScenarioInputs {
  const batchThreshold = getBatchThreshold({
    flopsPerByte: overrides.flopsPerByte ?? hardware.flopsPerByte,
    totalParams: model.totalParams,
    activeParams: model.activeParams,
  });

  return {
    hardware,
    model,
    contextTokens: overrides.contextTokens ?? model.contextTokens,
    batchSize: overrides.batchSize ?? Math.max(1, Math.round(batchThreshold)),
    weightBytesPerParam: overrides.weightBytesPerParam ?? model.defaultWeightBytesPerParam,
    kvBytesPerToken: overrides.kvBytesPerToken ?? model.kvBytesPerToken,
    flopsPerByte: overrides.flopsPerByte ?? hardware.flopsPerByte,
    tokensPerSecond: overrides.tokensPerSecond ?? 50e6,
    desiredConcurrentUsers: overrides.desiredConcurrentUsers ?? 128,
    deploymentDays: overrides.deploymentDays ?? 60,
    pipelineStages: overrides.pipelineStages ?? 1,
    expertParallelism: overrides.expertParallelism ?? hardware.gpuCount,
    safetyMargin: overrides.safetyMargin ?? 0.8,
  };
}

export function calculateScenario(input: ScenarioInputs): ScenarioResult {
  const isDecoder = decoderArchitectures.has(input.model.architecture);
  const batchThreshold = getBatchThreshold({
    flopsPerByte: input.flopsPerByte,
    totalParams: input.model.totalParams,
    activeParams: input.model.activeParams,
  });
  const hbmDrainSeconds = getHbmDrainTime({
    memoryBytes: input.hardware.memoryBytesPerGpu,
    bandwidthBytesPerSecond: input.hardware.memoryBandwidthBytesPerSecondPerGpu,
  });
  const weightBytesTotal = getWeightFootprint({
    totalParams: input.model.totalParams,
    weightBytesPerParam: input.weightBytesPerParam,
  });
  const kvBytesTotal = isDecoder
    ? getKvFootprint({
        batch: input.batchSize,
        contextTokens: input.contextTokens,
        kvBytesPerToken: input.kvBytesPerToken,
      })
    : 0;
  const requiredBytesPerGpu = getPipelineMemoryPerGpu({
    weightsBytes: weightBytesTotal,
    kvBytes: kvBytesTotal,
    expertParallelism: input.expertParallelism,
    pipelineStages: input.pipelineStages,
  });
  const availableBytesPerGpu = input.hardware.memoryBytesPerGpu * input.safetyMargin;
  const memoryUtilization = requiredBytesPerGpu / availableBytesPerGpu;
  const memoryRemainingBytesPerGpu = availableBytesPerGpu - requiredBytesPerGpu;
  const maxFittingBatch = isDecoder
    ? getMaxFittingBatch({
        availableBytesPerGpu,
        weightsBytes: weightBytesTotal,
        contextTokens: input.contextTokens,
        kvBytesPerToken: input.kvBytesPerToken,
        expertParallelism: input.expertParallelism,
        pipelineStages: input.pipelineStages,
      })
    : Number.NaN;
  const inferenceTokens = getInferenceTokens({
    tokensPerSecond: input.tokensPerSecond,
    deploymentDays: input.deploymentDays,
  });
  const chinchillaRatio = getChinchillaRatio({
    inferenceTokens,
    activeParams: input.model.activeParams,
  });
  const bottleneck = getBottleneck(input, weightBytesTotal, kvBytesTotal);

  const verdict = getVerdict(input.model.architecture, memoryUtilization);
  const warnings = getWarnings(input, memoryUtilization, batchThreshold);
  const { verdictLabel, mainReason, nextAction } = explainResult(input, verdict, bottleneck, memoryUtilization);

  return {
    verdict,
    verdictLabel,
    mainReason,
    nextAction,
    warnings,
    batchThreshold,
    hbmDrainSeconds,
    weightBytesTotal,
    kvBytesTotal,
    requiredBytesPerGpu,
    availableBytesPerGpu,
    memoryUtilization,
    memoryRemainingBytesPerGpu,
    maxFittingBatch,
    inferenceTokens,
    chinchillaRatio,
    bottleneck,
  };
}

export function getBatchThreshold({
  flopsPerByte,
  totalParams,
  activeParams,
}: {
  flopsPerByte: number;
  totalParams: number;
  activeParams: number;
}): number {
  return safeDivide(flopsPerByte * totalParams, activeParams);
}

export function getHbmDrainTime({
  memoryBytes,
  bandwidthBytesPerSecond,
}: {
  memoryBytes: number;
  bandwidthBytesPerSecond: number;
}): number {
  return safeDivide(memoryBytes, bandwidthBytesPerSecond);
}

export function getWeightFootprint({
  totalParams,
  weightBytesPerParam,
}: {
  totalParams: number;
  weightBytesPerParam: number;
}): number {
  return totalParams * weightBytesPerParam;
}

export function getKvFootprint({
  batch,
  contextTokens,
  kvBytesPerToken,
}: {
  batch: number;
  contextTokens: number;
  kvBytesPerToken: number;
}): number {
  return batch * contextTokens * kvBytesPerToken;
}

export function getPipelineMemoryPerGpu({
  weightsBytes,
  kvBytes,
  expertParallelism,
  pipelineStages,
}: {
  weightsBytes: number;
  kvBytes: number;
  expertParallelism: number;
  pipelineStages: number;
}): number {
  const denominator = Math.max(1, expertParallelism * pipelineStages);
  return (weightsBytes + kvBytes) / denominator;
}

export function getInferenceTokens({
  tokensPerSecond,
  deploymentDays,
}: {
  tokensPerSecond: number;
  deploymentDays: number;
}): number {
  return tokensPerSecond * deploymentDays * 86400;
}

export function getChinchillaRatio({
  inferenceTokens,
  activeParams,
}: {
  inferenceTokens: number;
  activeParams: number;
}): number {
  return safeDivide(inferenceTokens, 20 * activeParams);
}

export function createServingPlan(input: ScenarioInputs, result: ScenarioResult): ServingPlan {
  const requestedConcurrency = Math.max(1, Math.floor(input.desiredConcurrentUsers));
  const maxFittingConcurrency = Number.isFinite(result.maxFittingBatch) ? Math.max(0, result.maxFittingBatch) : 0;
  const fitsRequestedConcurrency = requestedConcurrency <= maxFittingConcurrency && result.verdict !== "not-applicable";
  const plannedConcurrency = Math.max(1, Math.min(requestedConcurrency, Math.max(1, maxFittingConcurrency)));
  const recommendedTensorParallelSize = Math.max(1, input.hardware.gpuCount);
  const recommendedGpuMemoryUtilization = clamp(input.safetyMargin, 0.5, 0.95);
  const recommendedMaxModelLen = Math.floor(input.contextTokens);
  const recommendedMaxNumSeqs = plannedConcurrency;
  const recommendedMaxNumBatchedTokens = Math.max(
    recommendedMaxModelLen,
    Math.min(recommendedMaxModelLen * recommendedMaxNumSeqs, 1_048_576),
  );
  const warnings: string[] = [];

  if (!fitsRequestedConcurrency) {
    warnings.push(
      `Requested concurrency ${requestedConcurrency} is above the estimated fitting concurrency ${maxFittingConcurrency}.`,
    );
  }
  if (input.hardware.interconnect === "multi-node-network") {
    warnings.push("Tensor parallel across multiple servers is usually not the same as one NVLink domain; validate networking.");
  }
  if (result.batchThreshold > plannedConcurrency * 4) {
    warnings.push(
      `The cost-efficient batch threshold is ${Math.round(result.batchThreshold)}, far above planned concurrency ${plannedConcurrency}. Expect lower throughput efficiency.`,
    );
  }
  if (input.model.kvConfidence !== "source-backed") {
    warnings.push("KV cache size is estimated; validate vLLM memory profiling before committing this config.");
  }

  const flags = [
    `--tensor-parallel-size ${recommendedTensorParallelSize}`,
    `--max-model-len ${recommendedMaxModelLen}`,
    `--gpu-memory-utilization ${recommendedGpuMemoryUtilization.toFixed(2)}`,
    `--max-num-seqs ${recommendedMaxNumSeqs}`,
    `--max-num-batched-tokens ${recommendedMaxNumBatchedTokens}`,
    "--enable-prefix-caching",
  ];

  return {
    requestedConcurrency,
    plannedConcurrency,
    fitsRequestedConcurrency,
    maxFittingConcurrency,
    concurrencyHeadroom: maxFittingConcurrency - requestedConcurrency,
    recommendedTensorParallelSize,
    recommendedGpuMemoryUtilization,
    recommendedMaxModelLen,
    recommendedMaxNumSeqs,
    recommendedMaxNumBatchedTokens,
    recommendedFlags: flags,
    command: [`vllm serve ${input.model.label}`, ...flags.map((flag) => `  ${flag}`)].join(" \\\n"),
    summary: fitsRequestedConcurrency
      ? `Plan for ${plannedConcurrency} concurrent users at ${recommendedMaxModelLen} tokens context.`
      : `Requested ${requestedConcurrency} concurrent users does not fit; planning caps at ${plannedConcurrency}.`,
    warnings,
  };
}

export function getMaxFittingBatch({
  availableBytesPerGpu,
  weightsBytes,
  contextTokens,
  kvBytesPerToken,
  expertParallelism,
  pipelineStages,
}: {
  availableBytesPerGpu: number;
  weightsBytes: number;
  contextTokens: number;
  kvBytesPerToken: number;
  expertParallelism: number;
  pipelineStages: number;
}): number {
  const totalAvailableBytes = availableBytesPerGpu * Math.max(1, expertParallelism * pipelineStages);
  const bytesAvailableForKv = totalAvailableBytes - weightsBytes;
  if (bytesAvailableForKv <= 0 || contextTokens <= 0 || kvBytesPerToken <= 0) return 0;
  return Math.floor(bytesAvailableForKv / (contextTokens * kvBytesPerToken));
}

function getVerdict(architecture: string, memoryUtilization: number): Verdict {
  if (!decoderArchitectures.has(architecture)) return "not-applicable";
  if (memoryUtilization <= 0.8) return "fits";
  if (memoryUtilization <= 1) return "tight";
  return "does-not-fit";
}

function getBottleneck(
  input: ScenarioInputs,
  weightBytesTotal: number,
  kvBytesTotal: number,
): ScenarioResult["bottleneck"] {
  if (!decoderArchitectures.has(input.model.architecture)) return "not-applicable";
  const computePressure = input.batchSize * input.model.activeParams;
  const memoryPressure = weightBytesTotal + kvBytesTotal;
  if (kvBytesTotal > weightBytesTotal * 1.4) return "kv-memory";
  if (memoryPressure > computePressure / Math.max(1, input.flopsPerByte)) return "weight-memory";
  return "compute";
}

function explainResult(
  input: ScenarioInputs,
  verdict: Verdict,
  bottleneck: ScenarioResult["bottleneck"],
  memoryUtilization: number,
): Pick<ScenarioResult, "verdictLabel" | "mainReason" | "nextAction"> {
  if (verdict === "not-applicable") {
    return {
      verdictLabel: "Not applicable for decoder guidance",
      mainReason: `${input.model.label} is a ${input.model.architecture} workload, so standard decode/KV-cache guidance is not the right model.`,
      nextAction: "Use the size fields as rough capacity hints, but benchmark this workload directly.",
    };
  }

  if (verdict === "does-not-fit") {
    return {
      verdictLabel: `Does not fit on ${input.hardware.label}`,
      mainReason: `Estimated per-GPU memory is ${(memoryUtilization * 100).toFixed(0)}% of the safety-adjusted HBM budget.`,
      nextAction: "Try lower context, smaller batch, KV quantization, lower weight precision, or a larger hardware preset.",
    };
  }

  if (verdict === "tight") {
    return {
      verdictLabel: `Tight fit on ${input.hardware.label}`,
      mainReason: `This fits only inside the safety margin; ${bottleneckLabel(bottleneck)} is the likely pressure point.`,
      nextAction: "Keep extra headroom for runtime overhead, routing imbalance, and cache fragmentation.",
    };
  }

  return {
    verdictLabel: `Fits on ${input.hardware.label}`,
    mainReason: `${bottleneckLabel(bottleneck)} is the main thing to watch, not raw parameter count alone.`,
    nextAction:
      input.hardware.interconnect === "multi-node-network"
        ? "Validate cross-server bandwidth before assuming this behaves like one large scale-up domain."
        : "Use the context and batch sliders to find the largest comfortable operating point.",
  };
}

function getWarnings(input: ScenarioInputs, memoryUtilization: number, batchThreshold: number): string[] {
  const warnings: string[] = [];
  if (input.hardware.interconnect === "multi-node-network") {
    warnings.push("This hardware pool spans multiple servers; network bandwidth and latency can dominate MoE or pipeline traffic.");
  }
  if (input.model.kvConfidence !== "source-backed") {
    warnings.push("KV bytes per token is estimated. Real values depend on attention type, KV quantization, and runtime implementation.");
  }
  if (!decoderArchitectures.has(input.model.architecture)) {
    warnings.push("This is not a standard decoder LLM workload; decode and KV-cache estimates are not directly meaningful.");
  }
  if (input.contextTokens >= 500000) {
    warnings.push("Long context is selected; KV cache is likely to dominate memory and bandwidth.");
  }
  if (input.batchSize < batchThreshold * 0.5) {
    warnings.push("Selected batch is far below the weight-amortization threshold, so per-token cost may be high.");
  }
  if (memoryUtilization > 0.9 && memoryUtilization <= 1) {
    warnings.push("Memory fit is tight; runtime overhead may push this over the edge.");
  }
  return warnings;
}

function bottleneckLabel(bottleneck: ScenarioResult["bottleneck"]): string {
  if (bottleneck === "kv-memory") return "KV cache memory";
  if (bottleneck === "weight-memory") return "weight memory bandwidth";
  if (bottleneck === "compute") return "compute";
  return "workload shape";
}

function safeDivide(numerator: number, denominator: number): number {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) return Number.NaN;
  return numerator / denominator;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
