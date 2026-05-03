export type Architecture = "dense" | "moe" | "embedding" | "vlm" | "hybrid";
export type Confidence = "source-backed" | "estimated" | "user-provided" | "unknown";
export type Interconnect = "single-node-nvlink" | "multi-node-network" | "rack-nvl" | "single-gpu";
export type PrecisionMode = "bf16" | "fp8" | "fp4" | "custom";
export type Verdict = "fits" | "tight" | "does-not-fit" | "not-applicable";

export type HardwarePreset = {
  id: string;
  label: string;
  gpuCount: number;
  memoryBytesPerGpu: number;
  memoryBandwidthBytesPerSecondPerGpu: number;
  flopsPerByte: number;
  // Bytes-per-param at the precision flopsPerByte was measured at.
  // 0.5 = fp4 (Blackwell), 1 = fp8 (Hopper), 2 = bf16 (Ampere/Ada).
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
