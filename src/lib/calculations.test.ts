import { describe, expect, it } from "vitest";
import { getBatchThreshold, getHbmDrainTime, getMaxFittingBatch } from "./calculations";

describe("core calculations", () => {
  it("computes H200 HBM drain time", () => {
    expect(getHbmDrainTime({ memoryBytes: 141e9, bandwidthBytesPerSecond: 4.8e12 })).toBeCloseTo(0.029375);
  });

  it("computes DeepSeek-style batch threshold", () => {
    expect(getBatchThreshold({ flopsPerByte: 300, totalParams: 671e9, activeParams: 37e9 })).toBeCloseTo(5440.54);
  });

  it("computes Qwen3-Coder-Next threshold on B300 ratio", () => {
    expect(getBatchThreshold({ flopsPerByte: 1875, totalParams: 80e9, activeParams: 3e9 })).toBeCloseTo(50000);
  });

  it("computes max fitting batch separately from the weight-amortization threshold", () => {
    expect(
      getMaxFittingBatch({
        availableBytesPerGpu: 288e9 * 0.8,
        weightsBytes: 80e9 * 0.5,
        contextTokens: 262144,
        kvBytesPerToken: 24576,
        expertParallelism: 8,
        pipelineStages: 1,
      }),
    ).toBe(279);
  });
});
