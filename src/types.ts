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
  interconnect: Interconnect;
  confidence: Confidence;
  notes: string;
  sources: string[];
};

export type ModelPreset = {
  id: string;
  label: string;
  litellmName?: string;
  architecture: Architecture;
  totalParams: number;
  activeParams: number;
  contextTokens: number;
  defaultWeightBytesPerParam: number;
  kvBytesPerToken: number;
  confidence: Confidence;
  kvConfidence: Confidence;
  notes: string;
  sources: string[];
};

export type ScenarioInputs = {
  hardware: HardwarePreset;
  model: ModelPreset;
  contextTokens: number;
  batchSize: number;
  weightBytesPerParam: number;
  kvBytesPerToken: number;
  flopsPerByte: number;
  tokensPerSecond: number;
  desiredConcurrentUsers: number;
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
  requestedConcurrency: number;
  plannedConcurrency: number;
  fitsRequestedConcurrency: boolean;
  maxFittingConcurrency: number;
  concurrencyHeadroom: number;
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
