import type { PrecisionMode } from "../types";

// Query-string codec for sharing a planner scenario as a plain URL.
// Deliberately the query string, NOT the hash — the Docs page uses #anchors.
// Flow: the sender copies a URL built by encodeShareParams; on load the app
// (before React mounts) decodes the params, writes them into localStorage —
// the same store the App's useLocalStorageState initializers read
// synchronously — and strips the query with history.replaceState so
// localStorage owns the state from then on.

export const STORAGE_PREFIX = "dwarkoptimus.";

export type SharedScenario = {
  hardwareId?: string;
  modelId?: string;
  precision?: PrecisionMode;
  customWeightBytes?: number;
  contextTokens?: number;
  batchSize?: number;
  autoTuneBatch?: boolean;
  kvBytesPerToken?: number;
  kCacheType?: string;
  vCacheType?: string;
  pipelineStages?: number;
  expertParallelism?: number;
  safetyMargin?: number;
  deploymentDays?: number;
  tokensPerSecond?: number;
  costPerGpuHour?: number;
};

// Short keys keep shared URLs readable.
const paramKeys = {
  hardwareId: "hw",
  modelId: "m",
  precision: "p",
  customWeightBytes: "wb",
  contextTokens: "ctx",
  batchSize: "b",
  autoTuneBatch: "at",
  kvBytesPerToken: "kv",
  kCacheType: "kq",
  vCacheType: "vq",
  pipelineStages: "pp",
  expertParallelism: "ep",
  safetyMargin: "sm",
  deploymentDays: "d",
  tokensPerSecond: "tps",
  costPerGpuHour: "usd",
} as const;

const precisionValues = new Set<string>(["bf16", "fp8", "fp4", "custom"]);

export function encodeShareParams(scenario: SharedScenario): URLSearchParams {
  const params = new URLSearchParams();
  const set = (key: string, value: string | number | boolean | undefined) => {
    if (value === undefined) return;
    params.set(key, String(value));
  };
  set(paramKeys.hardwareId, scenario.hardwareId);
  set(paramKeys.modelId, scenario.modelId);
  set(paramKeys.precision, scenario.precision);
  set(paramKeys.customWeightBytes, scenario.customWeightBytes);
  set(paramKeys.contextTokens, scenario.contextTokens);
  set(paramKeys.batchSize, scenario.batchSize);
  set(paramKeys.autoTuneBatch, scenario.autoTuneBatch === undefined ? undefined : scenario.autoTuneBatch ? 1 : 0);
  set(paramKeys.kvBytesPerToken, scenario.kvBytesPerToken);
  set(paramKeys.kCacheType, scenario.kCacheType);
  set(paramKeys.vCacheType, scenario.vCacheType);
  set(paramKeys.pipelineStages, scenario.pipelineStages);
  set(paramKeys.expertParallelism, scenario.expertParallelism);
  set(paramKeys.safetyMargin, scenario.safetyMargin);
  set(paramKeys.deploymentDays, scenario.deploymentDays);
  set(paramKeys.tokensPerSecond, scenario.tokensPerSecond);
  set(paramKeys.costPerGpuHour, scenario.costPerGpuHour);
  return params;
}

export function hasShareParams(params: URLSearchParams): boolean {
  return Object.values(paramKeys).some((key) => params.has(key));
}

export function decodeShareParams(params: URLSearchParams): SharedScenario {
  const out: SharedScenario = {};
  const readNumber = (key: string): number | undefined => {
    const raw = params.get(key);
    if (raw === null) return undefined;
    const value = Number(raw);
    return Number.isFinite(value) ? value : undefined;
  };
  const readString = (key: string): string | undefined => params.get(key) ?? undefined;

  out.hardwareId = readString(paramKeys.hardwareId);
  out.modelId = readString(paramKeys.modelId);
  const precision = readString(paramKeys.precision);
  if (precision !== undefined && precisionValues.has(precision)) out.precision = precision as PrecisionMode;
  out.customWeightBytes = readNumber(paramKeys.customWeightBytes);
  out.contextTokens = readNumber(paramKeys.contextTokens);
  out.batchSize = readNumber(paramKeys.batchSize);
  const autoTune = readString(paramKeys.autoTuneBatch);
  if (autoTune !== undefined) out.autoTuneBatch = autoTune === "1" || autoTune === "true";
  out.kvBytesPerToken = readNumber(paramKeys.kvBytesPerToken);
  out.kCacheType = readString(paramKeys.kCacheType);
  out.vCacheType = readString(paramKeys.vCacheType);
  out.pipelineStages = readNumber(paramKeys.pipelineStages);
  out.expertParallelism = readNumber(paramKeys.expertParallelism);
  out.safetyMargin = readNumber(paramKeys.safetyMargin);
  out.deploymentDays = readNumber(paramKeys.deploymentDays);
  out.tokensPerSecond = readNumber(paramKeys.tokensPerSecond);
  out.costPerGpuHour = readNumber(paramKeys.costPerGpuHour);
  return out;
}

type StorageLike = Pick<Storage, "getItem" | "setItem">;

export type ApplyShareOptions = {
  knownHardwareIds: Set<string>;
  // Curated preset ids plus whatever HF imports the RECIPIENT has cached —
  // a shared HF model the recipient never imported can't be reconstructed
  // from the URL, so we leave their model selection alone.
  knownModelIds: Set<string>;
  defaultEnabledHardwareIds: string[];
};

// Writes the shared scenario into localStorage under the same keys the App's
// useLocalStorageState hooks read on mount. Returns true if anything was
// applied. Values are JSON-encoded to match useLocalStorageState.
export function applySharedScenario(
  shared: SharedScenario,
  storage: StorageLike,
  options: ApplyShareOptions,
): boolean {
  let applied = false;
  const write = (key: string, value: unknown) => {
    storage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
    applied = true;
  };

  if (shared.hardwareId !== undefined && options.knownHardwareIds.has(shared.hardwareId)) {
    write("hardwareId", shared.hardwareId);
    // Make sure the shared hardware is actually enabled for the recipient,
    // otherwise the Planner silently falls back to their first enabled preset.
    let enabled: string[] = options.defaultEnabledHardwareIds;
    try {
      const raw = storage.getItem(STORAGE_PREFIX + "enabledHardwareIds");
      if (raw !== null) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) enabled = parsed.filter((id): id is string => typeof id === "string");
      }
    } catch {
      // corrupted storage — fall back to defaults
    }
    if (!enabled.includes(shared.hardwareId)) {
      write("enabledHardwareIds", [...enabled, shared.hardwareId]);
    }
  }
  if (shared.modelId !== undefined && options.knownModelIds.has(shared.modelId)) {
    write("modelId", shared.modelId);
  }
  if (shared.precision !== undefined) write("precision", shared.precision);
  if (shared.customWeightBytes !== undefined) write("customWeightBytes", shared.customWeightBytes);
  if (shared.contextTokens !== undefined) write("contextTokens", shared.contextTokens);
  if (shared.batchSize !== undefined) write("batchSize", shared.batchSize);
  if (shared.autoTuneBatch !== undefined) write("autoTuneBatch", shared.autoTuneBatch);
  if (shared.kvBytesPerToken !== undefined) write("kvBytesPerToken", shared.kvBytesPerToken);
  if (shared.kCacheType !== undefined) write("kCacheType", shared.kCacheType);
  if (shared.vCacheType !== undefined) write("vCacheType", shared.vCacheType);
  if (shared.pipelineStages !== undefined) write("pipelineStages", shared.pipelineStages);
  if (shared.expertParallelism !== undefined) write("expertParallelism", shared.expertParallelism);
  if (shared.safetyMargin !== undefined) write("safetyMargin", shared.safetyMargin);
  if (shared.deploymentDays !== undefined) write("deploymentDays", shared.deploymentDays);
  if (shared.tokensPerSecond !== undefined) write("tokensPerSecond", shared.tokensPerSecond);
  if (shared.costPerGpuHour !== undefined) write("costPerGpuHour", shared.costPerGpuHour);
  if (applied) write("activeTab", "planner");
  return applied;
}
