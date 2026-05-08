export type Architecture = "dense" | "moe" | "embedding" | "vlm" | "hybrid";
export type Confidence = "source-backed" | "estimated" | "user-provided" | "unknown";
export type Interconnect = "single-node-nvlink" | "multi-node-network" | "rack-nvl" | "single-gpu";
export type PrecisionMode = "bf16" | "fp8" | "fp4" | "custom";
export type Verdict = "fits" | "tight" | "does-not-fit" | "not-applicable";
export type HardwareCategory =
  | "nvidia-blackwell"
  | "nvidia-hopper"
  | "nvidia-ampere"
  | "nvidia-consumer"
  | "nvidia-legacy"
  | "apple-silicon";

export type HardwarePreset = {
  id: string;
  label: string;
  category: HardwareCategory;
  gpuCount: number;
  memoryBytesPerGpu: number;
  memoryBandwidthBytesPerSecondPerGpu: number;
  flopsPerByte: number;
  // Bytes-per-param at the precision flopsPerByte was measured at.
  // 0.5 = fp4 (Blackwell), 1 = fp8 (Hopper), 2 = bf16 (Ampere/Ada/Apple).
  // Used to rescale roofline math when serving at a different precision than the spec.
  nativeComputeBytes: number;
  interconnect: Interconnect;
  confidence: Confidence;
  notes: string;
  sources: string[];
};

export type ModelPreset = {
  id: string;
  label: string;
  architecture: Architecture;
  totalParams: number;
  activeParams: number;
  contextTokens: number;
  defaultWeightBytesPerParam: number;
  kvBytesPerToken: number;
  // Number of experts a single token is routed to (MoE) — used to estimate
  // scale-up vs scale-out traffic when the deployment crosses racks.
  activatedExperts?: number;
  // Approximate transformer layer count. Used for the MoE multi-rack
  // feasibility check; treated as a rough estimate.
  approxLayers?: number;
  confidence: Confidence;
  kvConfidence: Confidence;
  notes: string;
  sources: string[];
};

export type ScenarioInputs = {
  hardware: HardwarePreset;
  model: ModelPreset;
  contextTokens: number;
  // Number of concurrent sequences kept in flight per decode step. This is
  // the "B" from the roofline lecture and the only concurrency knob.
  batchSize: number;
  weightBytesPerParam: number;
  kvBytesPerToken: number;
  flopsPerByte: number;
  tokensPerSecond: number;
  deploymentDays: number;
  pipelineStages: number;
  expertParallelism: number;
  safetyMargin: number;
  // Lifecycle FLOPs accounting (Reiner Pope's three-phase model). All default
  // to 0 — the lifecycle panel is only meaningful once pretrainTokens > 0.
  pretrainTokens: number;
  rlTokens: number;
  // Multiplier on the 2N forward baseline for RL FLOPs. Default 3 ≈ matches
  // the 6N pretrain coefficient (forward + backward + RL overhead).
  rlInefficiency: number;
  // Multiplier on the 2N forward baseline for inference FLOPs. Default 5
  // captures the lecture's "decode MFU ≈ 1/5 of prefill" rule of thumb.
  inferenceInefficiency: number;
};

export type LifecycleFlops = {
  pretrainFlops: number;
  rlFlops: number;
  inferenceFlops: number;
  totalFlops: number;
  pretrainShare: number;
  rlShare: number;
  inferenceShare: number;
  // Which phase is the largest, and the ratio between the largest and the
  // second-largest (1 = perfectly balanced; 10 = one phase is 10× the next).
  dominantPhase: "pretrain" | "rl" | "inference" | "none";
  dominanceRatio: number;
};

export type ScenarioResult = {
  verdict: Verdict;
  verdictLabel: string;
  mainReason: string;
  nextAction: string;
  warnings: string[];
  batchThreshold: number;
  hbmDrainSeconds: number;
  stepIntervalSeconds: number;
  derivedTokensPerSecond: number;
  sparsityRatio: number;
  weightBytesTotal: number;
  kvBytesTotal: number;
  requiredBytesPerGpu: number;
  availableBytesPerGpu: number;
  memoryUtilization: number;
  memoryRemainingBytesPerGpu: number;
  maxFittingBatch: number;
  inferenceTokens: number;
  chinchillaRatio: number;
  bottleneck: "compute" | "weight-memory" | "kv-memory" | "not-applicable";
  // Prefill vs decode efficiency (Reiner's "decode is memory-bound, prefill
  // is compute-bound" framing). prefillTokensPerSecond is the asymptotic
  // compute-bound rate; decodeMfu is the fraction of step time actually
  // spent doing FLOPs (1 - memory waste). NaN for non-decoder workloads.
  prefillTokensPerSecond: number;
  decodeMfu: number;
  // Context length at which per-token KV-fetch time equals per-token compute
  // time at this batch and active-param size — the boundary between the
  // compute-bound and memory-bound regimes. Below: compute is the wall.
  // Above: KV bandwidth is. NaN if kvBytesPerToken is 0.
  crossoverContextTokens: number;
  // Pipeline parallelism efficiency. With P stages and B in-flight micro-
  // batches, the bubble fraction is (P-1)/(B+P-1). pipelineEfficiency =
  // 1 - bubbleFraction is the throughput multiplier already applied to
  // derivedTokensPerSecond. Both = 1 / 0 when pipelineStages = 1.
  pipelineBubbleFraction: number;
  pipelineEfficiency: number;
  microBatchCount: number;
  // Three-phase lifecycle FLOPs (pretrain / RL / inference). All zero when
  // pretrainTokens = 0; the lifecycle panel only appears once a value is set.
  lifecycle: LifecycleFlops;
};

export type ServingPlan = {
  requestedBatch: number;
  plannedBatch: number;
  fitsRequestedBatch: boolean;
  maxFittingBatch: number;
  batchHeadroom: number;
  recommendedTensorParallelSize: number;
  recommendedGpuMemoryUtilization: number;
  recommendedMaxModelLen: number;
  recommendedMaxNumSeqs: number;
  recommendedMaxNumBatchedTokens: number;
  recommendedFlags: string[];
  command: string;
  summary: string;
  warnings: string[];
};
