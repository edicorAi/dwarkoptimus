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
    weightBytesPerParam: overrides.weightBytesPerParam ?? model.defaultWeightBytesPerParam,
    nativeComputeBytes: hardware.nativeComputeBytes,
  });

  return {
    hardware,
    model,
    contextTokens: overrides.contextTokens ?? model.contextTokens,
    batchSize: overrides.batchSize ?? Math.max(1, Math.round(batchThreshold)),
    weightBytesPerParam: overrides.weightBytesPerParam ?? model.defaultWeightBytesPerParam,
    kvBytesPerToken: overrides.kvBytesPerToken ?? model.kvBytesPerToken,
    flopsPerByte: overrides.flopsPerByte ?? hardware.flopsPerByte,
    tokensPerSecond: overrides.tokensPerSecond ?? 0,
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
    weightBytesPerParam: input.weightBytesPerParam,
    nativeComputeBytes: input.hardware.nativeComputeBytes,
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
  // Reiner Pope rule of thumb: the natural step interval equals the HBM drain
  // time (capacity ÷ bandwidth). Faster is physically impossible (you can't
  // read all weights in less time than bandwidth allows); slower means FLOPs
  // sit idle. Pool throughput = batch / step, so the user sees a derived
  // tokens/sec rather than having to guess.
  const stepIntervalSeconds = isDecoder ? hbmDrainSeconds : Number.NaN;
  const derivedTokensPerSecond =
    isDecoder && stepIntervalSeconds > 0 ? input.batchSize / stepIntervalSeconds : Number.NaN;
  const sparsityRatio = safeDivide(input.model.totalParams, input.model.activeParams);
  // If the user left the manual tps override at the default, prefer the derived value
  // for the lifetime-tokens estimate so the Chinchilla figure is realistic.
  const tpsForLifetime =
    Number.isFinite(derivedTokensPerSecond) && input.tokensPerSecond <= 0
      ? derivedTokensPerSecond
      : input.tokensPerSecond;
  const inferenceTokens = getInferenceTokens({
    tokensPerSecond: tpsForLifetime,
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
    stepIntervalSeconds,
    derivedTokensPerSecond,
    sparsityRatio,
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
  weightBytesPerParam,
  nativeComputeBytes,
}: {
  flopsPerByte: number;
  totalParams: number;
  activeParams: number;
  // Optional precision adjustment. When the model's weight precision differs from the
  // precision flopsPerByte was measured at, peak FLOPs scale roughly linearly with bit
  // width on tensor cores (fp4 ≈ 2× fp8 ≈ 4× bf16). Threshold scales as
  // (nativeComputeBytes / weightBytesPerParam). If either is omitted, no scaling.
  weightBytesPerParam?: number;
  nativeComputeBytes?: number;
}): number {
  const precisionScale =
    weightBytesPerParam && weightBytesPerParam > 0 && nativeComputeBytes && nativeComputeBytes > 0
      ? nativeComputeBytes / weightBytesPerParam
      : 1;
  return safeDivide(flopsPerByte * precisionScale * totalParams, activeParams);
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

// Reiner Pope's all-to-all feasibility heuristic. The ratio
//   t_scaleUp / t_scaleOut = (activatedExperts × layersPerStage × 2) / 8
// must be ≥ 1 for an MoE deployment to keep working when it crosses rack
// boundaries. Below 1 means scale-out network is the bottleneck.
// Treat hardware, model, context, and weight precision as user-fixed inputs.
// Only tune the rest: batch, KV cache quantization, EP, PP, safety margin.
// Returns the model's default kvBytesPerToken unchanged unless KV quantization
// was needed to make the config fit.
export type AutoOptimizeInputs = {
  contextTokens: number;
  weightBytesPerParam: number;
  kvBytesPerToken: number;
  batchSize: number;
  expertParallelism: number;
  pipelineStages: number;
  safetyMargin: number;
};

export type AutoOptimizeOutput = {
  overrides: {
    kvBytesPerToken: number;
    batchSize: number;
    expertParallelism: number;
    pipelineStages: number;
    safetyMargin: number;
  };
  rationale: string[];
};

export function autoOptimize(
  hardware: HardwarePreset,
  model: ModelPreset,
  current: AutoOptimizeInputs,
): AutoOptimizeOutput {
  const rationale: string[] = [];
  const targetExpertParallelism = Math.max(1, hardware.gpuCount);
  const targetPipelineStages = 1;
  const targetSafetyMargin = 0.85;

  if (current.expertParallelism !== targetExpertParallelism) {
    rationale.push(
      `Set expert parallelism to ${targetExpertParallelism} so the full NVLink domain shares weights and KV.`,
    );
  }
  if (current.pipelineStages !== targetPipelineStages) {
    rationale.push("Reset pipeline stages to 1 — pipelining saves capacity but does not reduce step time.");
  }
  if (current.safetyMargin !== targetSafetyMargin) {
    rationale.push("Set safety margin to 0.85 (≈15% HBM headroom for runtime overhead and fragmentation).");
  }

  const isDecoder = decoderArchitectures.has(model.architecture);
  // User-fixed:
  const contextTokens = current.contextTokens;
  const weightBytesPerParam = current.weightBytesPerParam;
  // Tunable:
  let kvBytesPerToken = current.kvBytesPerToken;

  const trial = (kv: number) => {
    const scenario = buildScenario(hardware, model, {
      contextTokens,
      batchSize: 1,
      weightBytesPerParam,
      kvBytesPerToken: kv,
      expertParallelism: targetExpertParallelism,
      pipelineStages: targetPipelineStages,
      safetyMargin: targetSafetyMargin,
    });
    return calculateScenario(scenario);
  };

  let result = trial(kvBytesPerToken);

  // KV quantization fallback: only if the workload is decoder-style AND the
  // current config doesn't fit AND quantizing KV would actually help (i.e.
  // KV is the dominant pressure, not weights).
  if (isDecoder && result.maxFittingBatch < 1 && kvBytesPerToken > 0) {
    const weightsFitAlone = result.weightBytesTotal < result.availableBytesPerGpu * targetExpertParallelism;
    if (weightsFitAlone) {
      // Try fp8 KV cache (halve) first.
      const halved = Math.max(1, Math.round(kvBytesPerToken / 2));
      const halvedResult = trial(halved);
      if (halvedResult.maxFittingBatch >= 1) {
        kvBytesPerToken = halved;
        result = halvedResult;
        rationale.push(
          "Halve KV bytes/token by enabling fp8 KV cache (`--kv-cache-dtype fp8`) — KV was overflowing HBM at the current context.",
        );
      } else {
        // Try int4 KV (quarter) as a last resort.
        const quartered = Math.max(1, Math.round(kvBytesPerToken / 4));
        const quarteredResult = trial(quartered);
        if (quarteredResult.maxFittingBatch >= 1) {
          kvBytesPerToken = quartered;
          result = quarteredResult;
          rationale.push(
            "Quarter KV bytes/token (4-bit KV cache) — fp8 KV alone wasn't enough. Validate quality before production.",
          );
        }
      }
    }
  }

  // Pick the cost-optimal batch, capped by what fits.
  const knee = Math.max(1, Math.round(result.batchThreshold));
  const ceiling = Math.max(1, Math.floor(result.maxFittingBatch));
  const targetBatch = isDecoder ? Math.min(knee, ceiling) : 1;

  if (isDecoder && targetBatch !== current.batchSize) {
    if (knee > ceiling) {
      rationale.push(
        `Set batch to ${targetBatch.toLocaleString()} — the largest that fits HBM at this context. Break-even is higher (${knee.toLocaleString()}), so per-token cost stays elevated; only bigger hardware or smaller context can close the gap.`,
      );
    } else {
      rationale.push(
        `Set batch to ${targetBatch.toLocaleString()} — the break-even point where per-token cost stops dropping (HBM still has room above it).`,
      );
    }
  }

  // Final feasibility check: if it still doesn't fit even with KV quantization,
  // tell the operator to revisit their fixed inputs (context / precision / hardware).
  if (isDecoder && result.maxFittingBatch < 1) {
    rationale.push(
      `Even with KV quantization this configuration does not fit. Reduce context length, drop weight precision, or pick larger hardware — those are the inputs auto-optimize doesn't touch.`,
    );
  }

  if (rationale.length === 0) rationale.push("Settings already look optimal — no changes applied.");

  return {
    overrides: {
      kvBytesPerToken,
      batchSize: isDecoder ? targetBatch : current.batchSize,
      expertParallelism: targetExpertParallelism,
      pipelineStages: targetPipelineStages,
      safetyMargin: targetSafetyMargin,
    },
    rationale,
  };
}

export function getMoeMultiRackRatio({
  activatedExperts,
  layersPerStage,
  scaleUpVsScaleOut = 8,
}: {
  activatedExperts: number;
  layersPerStage: number;
  scaleUpVsScaleOut?: number;
}): number {
  if (activatedExperts <= 0 || layersPerStage <= 0) return Number.NaN;
  return (activatedExperts * layersPerStage * 2) / scaleUpVsScaleOut;
}

// Returns latency and cost-per-token across a sweep of batch sizes — used by
// both the latency chart and the cost-per-token chart. Cost is normalized
// (seconds per token × 1) so the chart shape is what matters; the absolute
// dollars depend on $/GPU·hour, which we don't model.
export type RooflinePoint = {
  batch: number;
  computeSeconds: number;
  weightFetchSeconds: number;
  kvFetchSeconds: number;
  memorySeconds: number;
  totalSeconds: number;
  costPerToken: number;
};

export function getRooflineSweep(
  input: ScenarioInputs,
  options: { samples?: number; maxBatch?: number } = {},
): RooflinePoint[] {
  const samples = Math.max(2, Math.floor(options.samples ?? 32));
  const peakFlopsPool =
    input.flopsPerByte * input.hardware.memoryBandwidthBytesPerSecondPerGpu * input.hardware.gpuCount;
  const bandwidthPool = input.hardware.memoryBandwidthBytesPerSecondPerGpu * input.hardware.gpuCount;
  const weightBytes = input.model.totalParams * input.weightBytesPerParam;
  const weightFetchSeconds = weightBytes / bandwidthPool;
  const maxBatch = Math.max(2, options.maxBatch ?? Math.max(input.batchSize * 1.5, 100));
  return Array.from({ length: samples }, (_, index) => {
    const batch = 1 + (index / (samples - 1)) * (maxBatch - 1);
    const kvFetchSeconds = (batch * input.contextTokens * input.kvBytesPerToken) / bandwidthPool;
    const memorySeconds = weightFetchSeconds + kvFetchSeconds;
    const computeSeconds = (2 * batch * input.model.activeParams) / peakFlopsPool;
    const totalSeconds = Math.max(memorySeconds, computeSeconds);
    return {
      batch,
      computeSeconds,
      weightFetchSeconds,
      kvFetchSeconds,
      memorySeconds,
      totalSeconds,
      costPerToken: totalSeconds / batch,
    };
  });
}

export function createServingPlan(input: ScenarioInputs, result: ScenarioResult): ServingPlan {
  const requestedBatch = Math.max(1, Math.floor(input.batchSize));
  const maxFittingBatch = Number.isFinite(result.maxFittingBatch) ? Math.max(0, result.maxFittingBatch) : 0;
  const fitsRequestedBatch = requestedBatch <= maxFittingBatch && result.verdict !== "not-applicable";
  const plannedBatch = Math.max(1, Math.min(requestedBatch, Math.max(1, maxFittingBatch)));
  const pipelineParallelSize = Math.max(1, Math.floor(input.pipelineStages));
  // vLLM's parallel layout is GPUs = TP * PP; expert parallelism is implicit within TP for MoE.
  const recommendedTensorParallelSize = Math.max(1, Math.floor(input.hardware.gpuCount / pipelineParallelSize));
  const recommendedGpuMemoryUtilization = clamp(input.safetyMargin, 0.5, 0.95);
  const recommendedMaxModelLen = Math.floor(input.contextTokens);
  const recommendedMaxNumSeqs = plannedBatch;
  const recommendedMaxNumBatchedTokens = Math.max(
    recommendedMaxModelLen,
    Math.min(recommendedMaxModelLen * recommendedMaxNumSeqs, 1_048_576),
  );
  const warnings: string[] = [];

  if (!fitsRequestedBatch) {
    warnings.push(
      `Selected batch ${requestedBatch} is above the largest batch that fits in HBM (${maxFittingBatch}). Drop context, lower batch, or use larger hardware.`,
    );
  }
  if (input.hardware.interconnect === "multi-node-network") {
    warnings.push("Tensor parallel across multiple servers is usually not the same as one NVLink domain; validate networking.");
  }
  if (result.batchThreshold > plannedBatch * 4) {
    warnings.push(
      `Break-even batch is ≈ ${Math.round(result.batchThreshold)}, far above the planned ${plannedBatch}. Per-token cost will be high until traffic catches up.`,
    );
  }
  if (input.model.kvConfidence !== "source-backed") {
    warnings.push("KV cache size is estimated; validate vLLM memory profiling before committing this config.");
  }

  const kvQuantized =
    input.model.kvBytesPerToken > 0 && input.kvBytesPerToken < input.model.kvBytesPerToken * 0.75;
  const flags = [
    `--tensor-parallel-size ${recommendedTensorParallelSize}`,
    ...(pipelineParallelSize > 1 ? [`--pipeline-parallel-size ${pipelineParallelSize}`] : []),
    `--max-model-len ${recommendedMaxModelLen}`,
    `--gpu-memory-utilization ${recommendedGpuMemoryUtilization.toFixed(2)}`,
    `--max-num-seqs ${recommendedMaxNumSeqs}`,
    `--max-num-batched-tokens ${recommendedMaxNumBatchedTokens}`,
    ...(kvQuantized ? ["--kv-cache-dtype fp8"] : []),
    "--enable-prefix-caching",
  ];

  return {
    requestedBatch,
    plannedBatch,
    fitsRequestedBatch,
    maxFittingBatch,
    batchHeadroom: maxFittingBatch - requestedBatch,
    recommendedTensorParallelSize,
    recommendedGpuMemoryUtilization,
    recommendedMaxModelLen,
    recommendedMaxNumSeqs,
    recommendedMaxNumBatchedTokens,
    recommendedFlags: flags,
    command: [`vllm serve ${input.model.label}`, ...flags.map((flag) => `  ${flag}`)].join(" \\\n"),
    summary: fitsRequestedBatch
      ? `Plan for batch ${plannedBatch} at ${recommendedMaxModelLen} tokens context.`
      : `Selected batch ${requestedBatch} does not fit at this context — planning caps at ${plannedBatch}.`,
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
  // Roofline times per decode step:
  //   t_compute = 2 * activeParams * batch / FLOPs_peak
  //   t_memory  = (weightBytes + kvBytes) / bandwidth
  // memory-bound when t_memory > t_compute, i.e.
  //   flopsPerByte * (weightBytes + kvBytes) > 2 * activeParams * batch
  const memoryPressure = weightBytesTotal + kvBytesTotal;
  const computePressure = 2 * input.batchSize * input.model.activeParams;
  if (kvBytesTotal > weightBytesTotal * 1.4) return "kv-memory";
  if (memoryPressure * Math.max(1, input.flopsPerByte) > computePressure) return "weight-memory";
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
  // MoE on multi-node hardware: check the all-to-all feasibility ratio.
  if (
    input.hardware.interconnect === "multi-node-network" &&
    (input.model.architecture === "moe" || input.model.architecture === "hybrid") &&
    input.model.activatedExperts &&
    input.model.approxLayers
  ) {
    const layersPerStage = input.model.approxLayers / Math.max(1, input.pipelineStages);
    const ratio = getMoeMultiRackRatio({
      activatedExperts: input.model.activatedExperts,
      layersPerStage,
    });
    if (Number.isFinite(ratio) && ratio < 1) {
      warnings.push(
        `MoE all-to-all is likely network-bound on this multi-server pool (scale-up/scale-out ratio ≈ ${ratio.toFixed(2)} < 1). Run inside one NVLink domain or increase pipeline stages.`,
      );
    }
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
