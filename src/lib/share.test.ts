import { describe, expect, it } from "vitest";
import {
  applySharedScenario,
  decodeShareParams,
  encodeShareParams,
  hasShareParams,
  STORAGE_PREFIX,
  type SharedScenario,
} from "./share";

const fullScenario: SharedScenario = {
  hardwareId: "h200-server-4gpu",
  modelId: "qwen3-coder-next",
  precision: "fp8",
  customWeightBytes: 0.7,
  contextTokens: 131072,
  batchSize: 96,
  autoTuneBatch: false,
  kvBytesPerToken: 12288,
  kCacheType: "q8_0",
  vCacheType: "f16",
  pipelineStages: 2,
  expertParallelism: 4,
  safetyMargin: 0.85,
  deploymentDays: 90,
  tokensPerSecond: 25000,
  costPerGpuHour: 3.5,
};

function memoryStorage(initial: Record<string, string> = {}) {
  const store = new Map<string, string>(Object.entries(initial));
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    dump: () => Object.fromEntries(store),
  };
}

const options = {
  knownHardwareIds: new Set(["h200-server-4gpu", "dell-b300-8gpu"]),
  knownModelIds: new Set(["qwen3-coder-next", "kimi-k2.6"]),
  defaultEnabledHardwareIds: ["dell-b300-8gpu"],
};

describe("share codec", () => {
  it("round-trips every field", () => {
    const decoded = decodeShareParams(encodeShareParams(fullScenario));
    expect(decoded).toEqual(fullScenario);
  });

  it("round-trips a sparse scenario without inventing fields", () => {
    const sparse: SharedScenario = { hardwareId: "dell-b300-8gpu", contextTokens: 8192 };
    const decoded = decodeShareParams(encodeShareParams(sparse));
    expect(decoded.hardwareId).toBe("dell-b300-8gpu");
    expect(decoded.contextTokens).toBe(8192);
    expect(decoded.batchSize).toBeUndefined();
    expect(decoded.precision).toBeUndefined();
  });

  it("detects share params and ignores unrelated queries", () => {
    expect(hasShareParams(encodeShareParams(fullScenario))).toBe(true);
    expect(hasShareParams(new URLSearchParams("utm_source=x&foo=1"))).toBe(false);
  });

  it("drops malformed values instead of propagating them", () => {
    const params = new URLSearchParams("ctx=banana&p=fp99&b=64");
    const decoded = decodeShareParams(params);
    expect(decoded.contextTokens).toBeUndefined();
    expect(decoded.precision).toBeUndefined();
    expect(decoded.batchSize).toBe(64);
  });
});

describe("applying a shared scenario to storage", () => {
  it("writes JSON-encoded values under the app's storage keys", () => {
    const storage = memoryStorage();
    const applied = applySharedScenario(fullScenario, storage, options);
    expect(applied).toBe(true);
    expect(storage.getItem(`${STORAGE_PREFIX}hardwareId`)).toBe(JSON.stringify("h200-server-4gpu"));
    expect(storage.getItem(`${STORAGE_PREFIX}batchSize`)).toBe("96");
    expect(storage.getItem(`${STORAGE_PREFIX}autoTuneBatch`)).toBe("false");
    expect(storage.getItem(`${STORAGE_PREFIX}activeTab`)).toBe(JSON.stringify("planner"));
  });

  it("adds the shared hardware to the recipient's enabled inventory", () => {
    const storage = memoryStorage({
      [`${STORAGE_PREFIX}enabledHardwareIds`]: JSON.stringify(["dell-b300-8gpu"]),
    });
    applySharedScenario({ hardwareId: "h200-server-4gpu" }, storage, options);
    expect(JSON.parse(storage.getItem(`${STORAGE_PREFIX}enabledHardwareIds`)!)).toEqual([
      "dell-b300-8gpu",
      "h200-server-4gpu",
    ]);
  });

  it("seeds enabled hardware from the defaults when storage is empty", () => {
    const storage = memoryStorage();
    applySharedScenario({ hardwareId: "h200-server-4gpu" }, storage, options);
    expect(JSON.parse(storage.getItem(`${STORAGE_PREFIX}enabledHardwareIds`)!)).toEqual([
      "dell-b300-8gpu",
      "h200-server-4gpu",
    ]);
  });

  it("leaves the model untouched when the recipient doesn't know the id", () => {
    const storage = memoryStorage();
    applySharedScenario({ modelId: "hf:someone/private-model", contextTokens: 4096 }, storage, options);
    expect(storage.getItem(`${STORAGE_PREFIX}modelId`)).toBeNull();
    expect(storage.getItem(`${STORAGE_PREFIX}contextTokens`)).toBe("4096");
  });

  it("ignores unknown hardware ids entirely", () => {
    const storage = memoryStorage();
    applySharedScenario({ hardwareId: "gpu-from-the-future" }, storage, options);
    expect(storage.getItem(`${STORAGE_PREFIX}hardwareId`)).toBeNull();
    expect(storage.getItem(`${STORAGE_PREFIX}enabledHardwareIds`)).toBeNull();
  });
});
