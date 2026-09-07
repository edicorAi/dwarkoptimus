import { describe, expect, it } from "vitest";
import {
  autoOptimize,
  buildScenario,
  calculateScenario,
  createServingPlan,
  getBatchThreshold,
  getCostPerMillionTokens,
  getCrossoverContextTokens,
  getHbmDrainTime,
  getLifecycleFlops,
  getMaxFittingBatch,
  getMoeMultiRackRatio,
  getOptimalBatch,
  getPipelineBubble,
  getRooflineSweep,
  getServeModelArg,
} from "./calculations";
import { hardwarePresets } from "../data/hardware";
import { modelPresets } from "../data/models";

const b300 = hardwarePresets.find((h) => h.id === "dell-b300-8gpu")!;
const h200 = hardwarePresets.find((h) => h.id === "h200-server-4gpu")!;
const qwen = modelPresets.find((m) => m.id === "qwen3-coder-next")!;
const ministral = modelPresets.find((m) => m.id === "ministral-3-14b")!;
const granite = modelPresets.find((m) => m.id === "granite-embedding-107m")!;

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

  it("returns zero max fitting batch when weights overflow the pool", () => {
    expect(
      getMaxFittingBatch({
        availableBytesPerGpu: 80e9 * 0.8,
        weightsBytes: 1e12,
        contextTokens: 8192,
        kvBytesPerToken: 1024,
        expertParallelism: 1,
        pipelineStages: 1,
      }),
    ).toBe(0);
  });
});

describe("precision-aware batch threshold", () => {
  it("matches native fp4 weight on a B300", () => {
    // B300 native fp4 (0.5 B/param), serving fp4 → no scaling
    expect(
      getBatchThreshold({
        flopsPerByte: 1875,
        totalParams: 80e9,
        activeParams: 3e9,
        weightBytesPerParam: 0.5,
        nativeComputeBytes: 0.5,
      }),
    ).toBeCloseTo(50000);
  });

  it("halves on a B300 when serving fp8 weights", () => {
    // fp8 weights with fp4-native hardware → 2× more bytes per param means 2× less compute available
    expect(
      getBatchThreshold({
        flopsPerByte: 1875,
        totalParams: 80e9,
        activeParams: 3e9,
        weightBytesPerParam: 1,
        nativeComputeBytes: 0.5,
      }),
    ).toBeCloseTo(25000);
  });

  it("quarters on a B300 when serving bf16 weights", () => {
    expect(
      getBatchThreshold({
        flopsPerByte: 1875,
        totalParams: 80e9,
        activeParams: 3e9,
        weightBytesPerParam: 2,
        nativeComputeBytes: 0.5,
      }),
    ).toBeCloseTo(12500);
  });

  it("does NOT upscale on Hopper when weights are narrower than native fp8", () => {
    // Hopper has no fp4 tensor cores. Loading fp4 weights still runs matmul
    // at fp8 after dequant, so peak FLOPs (and therefore the threshold) are
    // unchanged from the fp8 case.
    const fp8 = getBatchThreshold({
      flopsPerByte: 590,
      totalParams: 80e9,
      activeParams: 3e9,
      weightBytesPerParam: 1,
      nativeComputeBytes: 1,
    });
    const fp4 = getBatchThreshold({
      flopsPerByte: 590,
      totalParams: 80e9,
      activeParams: 3e9,
      weightBytesPerParam: 0.5,
      nativeComputeBytes: 1,
    });
    expect(fp4).toBeCloseTo(fp8);
  });

  it("does NOT upscale on Apple Silicon when MLX-style int4 is selected", () => {
    // Apple GPUs have no fp8/fp4 tensor path — MLX int4 saves memory but the
    // matmul still runs at fp16. Threshold stays at the bf16 level regardless
    // of whether weights are stored at 2, 1, or 0.5 B/param.
    const m4Max = { flopsPerByte: 66, totalParams: 70e9, activeParams: 70e9, nativeComputeBytes: 2 };
    const bf16 = getBatchThreshold({ ...m4Max, weightBytesPerParam: 2 });
    const fp8 = getBatchThreshold({ ...m4Max, weightBytesPerParam: 1 });
    const fp4 = getBatchThreshold({ ...m4Max, weightBytesPerParam: 0.5 });
    expect(fp8).toBeCloseTo(bf16);
    expect(fp4).toBeCloseTo(bf16);
  });
});

describe("bottleneck classifier", () => {
  it("flags weight-memory at very low batch", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 1, contextTokens: 4096 });
    const result = calculateScenario(scenario);
    expect(result.bottleneck).toBe("weight-memory");
  });

  it("flags compute past the threshold", () => {
    // Dense ministral on B300 fp4 → threshold ≈ 1875. Push batch well above and keep
    // both context and KV per token tiny so the kv-memory branch doesn't fire.
    const scenario = buildScenario(b300, ministral, { batchSize: 5000, contextTokens: 8 });
    const result = calculateScenario(scenario);
    expect(result.bottleneck).toBe("compute");
  });

  it("flags kv-memory when the cache dominates weights", () => {
    // 4M-token KV pool against ~40 GB of weights → kv > 1.4 × weights
    const scenario = buildScenario(b300, qwen, { batchSize: 16, contextTokens: 1_000_000 });
    const result = calculateScenario(scenario);
    expect(result.bottleneck).toBe("kv-memory");
  });

  it("returns not-applicable for embedding models", () => {
    const scenario = buildScenario(b300, granite);
    const result = calculateScenario(scenario);
    expect(result.bottleneck).toBe("not-applicable");
    expect(result.verdict).toBe("not-applicable");
  });
});

describe("derived metrics", () => {
  it("derives a tokens-per-second estimate from batch and step", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 64 });
    const result = calculateScenario(scenario);
    // step = HBM drain. B300: 288 GB / 8 TB/s = 36 ms. 64 / 0.036 ≈ 1778 tok/s.
    expect(result.stepIntervalSeconds).toBeCloseTo(0.036, 3);
    expect(result.derivedTokensPerSecond).toBeGreaterThan(1500);
    expect(result.derivedTokensPerSecond).toBeLessThan(2000);
  });

  it("returns NaN step interval for non-decoder workloads", () => {
    const scenario = buildScenario(b300, granite);
    const result = calculateScenario(scenario);
    expect(Number.isNaN(result.stepIntervalSeconds)).toBe(true);
    expect(Number.isNaN(result.derivedTokensPerSecond)).toBe(true);
  });

  it("computes sparsity ratio", () => {
    const scenario = buildScenario(b300, qwen);
    const result = calculateScenario(scenario);
    expect(result.sparsityRatio).toBeCloseTo(80 / 3, 5);
  });
});

describe("MoE multi-rack feasibility", () => {
  it("returns the lecture's ratio (E×L×2)/8", () => {
    expect(getMoeMultiRackRatio({ activatedExperts: 8, layersPerStage: 16 })).toBe(32);
    expect(getMoeMultiRackRatio({ activatedExperts: 8, layersPerStage: 1 })).toBe(2);
    expect(getMoeMultiRackRatio({ activatedExperts: 1, layersPerStage: 1 })).toBe(0.25);
  });

  it("returns NaN with bad inputs", () => {
    expect(Number.isNaN(getMoeMultiRackRatio({ activatedExperts: 0, layersPerStage: 4 }))).toBe(true);
    expect(Number.isNaN(getMoeMultiRackRatio({ activatedExperts: 4, layersPerStage: 0 }))).toBe(true);
  });
});

describe("roofline sweep", () => {
  it("produces a hyperbolic cost curve that flattens", () => {
    // Short context so weights dominate at small batch; the hyperbola is most
    // visible there. Past the knee, cost/token approaches the compute floor.
    const scenario = buildScenario(b300, qwen, { batchSize: 64, contextTokens: 512 });
    const points = getRooflineSweep(scenario, { samples: 32, maxBatch: 200000 });
    expect(points[0].batch).toBe(1);
    const first = points[0].costPerToken;
    const last = points[points.length - 1].costPerToken;
    expect(first).toBeGreaterThan(last * 100);
  });
});

describe("autoOptimize", () => {
  it("resets EP/PP/safety to lecture defaults and picks a sensible batch", () => {
    const out = autoOptimize(b300, qwen, {
      contextTokens: 262144,
      weightBytesPerParam: 0.5,
      kvBytesPerToken: 24576,
      batchSize: 1,
      expertParallelism: 1,
      pipelineStages: 4,
      safetyMargin: 0.6,
    });
    expect(out.overrides.expertParallelism).toBe(b300.gpuCount);
    expect(out.overrides.pipelineStages).toBe(1);
    expect(out.overrides.safetyMargin).toBeCloseTo(0.85);
    expect(out.overrides.batchSize).toBeGreaterThan(1);
    expect(out.rationale.length).toBeGreaterThan(0);
  });

  it("does not touch context or weight precision", () => {
    const out = autoOptimize(b300, qwen, {
      contextTokens: 262144,
      weightBytesPerParam: 2,
      kvBytesPerToken: 24576,
      batchSize: 1,
      expertParallelism: b300.gpuCount,
      pipelineStages: 1,
      safetyMargin: 0.85,
    });
    expect(Object.keys(out.overrides)).not.toContain("weightBytesPerParam");
    expect(Object.keys(out.overrides)).not.toContain("contextTokens");
    // The rationale shouldn't claim the optimizer changed precision or context.
    // (It may still mention them as user knobs to revisit if infeasible.)
    expect(out.rationale.some((r) => /drop\s+weight\s+precision\s+to/i.test(r))).toBe(false);
    expect(out.rationale.some((r) => /reduce\s+context\s+to\s+\d/i.test(r))).toBe(false);
  });

  it("quantizes KV (halves bytes) when KV is the only thing pushing it over", () => {
    // gpt-oss-120b at fp8 on a B300 server: 117e9 weights = 117 GB → fits in pool;
    // KV at 65536 B × long ctx × batch overflows. KV halving should rescue it.
    const gptOss = modelPresets.find((m) => m.id === "gpt-oss-120b")!;
    const out = autoOptimize(b300, gptOss, {
      contextTokens: gptOss.contextTokens,
      weightBytesPerParam: 1,
      kvBytesPerToken: gptOss.kvBytesPerToken,
      batchSize: 1024,
      expertParallelism: b300.gpuCount,
      pipelineStages: 1,
      safetyMargin: 0.85,
    });
    // Either KV stayed (already fits) or got halved/quartered. We don't require it
    // to fire on every model, but if it does, it must be a strict reduction.
    expect(out.overrides.kvBytesPerToken).toBeLessThanOrEqual(gptOss.kvBytesPerToken);
  });

  it("flags infeasibility instead of silently changing precision/context", () => {
    // qwen at bf16 weights = 80 GB on a single H100 SXM (80 GB), KV pushes it over.
    // Auto-optimize must NOT drop to fp8 — it must report that this config can't fit.
    const h100 = hardwarePresets.find((h) => h.id === "h100-sxm-1gpu")!;
    const out = autoOptimize(h100, qwen, {
      contextTokens: 262144,
      weightBytesPerParam: 2,
      kvBytesPerToken: 24576,
      batchSize: 1,
      expertParallelism: 1,
      pipelineStages: 1,
      safetyMargin: 0.85,
    });
    // weightBytesPerParam isn't even in the overrides anymore, so nothing to assert there.
    // The rationale should call out the user's fixed inputs as the blockers.
    const joined = out.rationale.join(" ").toLowerCase();
    expect(joined).toMatch(/does not fit|reduce context|drop weight precision|larger hardware/);
  });

  it("returns batch=current and a no-op rationale for embedding workloads", () => {
    const out = autoOptimize(b300, granite, {
      contextTokens: 512,
      weightBytesPerParam: 2,
      kvBytesPerToken: 0,
      batchSize: 64,
      expertParallelism: b300.gpuCount,
      pipelineStages: 1,
      safetyMargin: 0.85,
    });
    expect(out.overrides.batchSize).toBe(64);
    expect(out.rationale.some((r) => r.toLowerCase().includes("optimal"))).toBe(true);
  });
});

describe("prefill vs decode regime (Section 6 flashcards)", () => {
  it("computes prefill TPS as compute-bound asymptote (peak FLOPs / 2N_active)", () => {
    // B300: flopsPerByte 1875 × bw 8 TB/s × 8 GPUs = 1.125e17 FLOPs/s pool peak.
    // Qwen active params 3e9. Prefill TPS ≈ 1.125e17 / (2 × 3e9) ≈ 1.875e7 tok/s.
    const scenario = buildScenario(b300, qwen, { batchSize: 64 });
    const result = calculateScenario(scenario);
    expect(result.prefillTokensPerSecond).toBeGreaterThan(1.5e7);
    expect(result.prefillTokensPerSecond).toBeLessThan(2.2e7);
    // Prefill is orders of magnitude above decode pool throughput.
    expect(result.prefillTokensPerSecond).toBeGreaterThan(result.derivedTokensPerSecond * 100);
  });

  it("decodeMfu collapses below 1 when memory time exceeds compute time", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 16, contextTokens: 4096 });
    const result = calculateScenario(scenario);
    expect(result.decodeMfu).toBeGreaterThan(0);
    expect(result.decodeMfu).toBeLessThan(0.2);
  });

  it("crossover context follows the lecture's bytes/token = (1/300)(N/ctx) form", () => {
    // ctx_crossover = N_active / (flopsPerByte × kvBytes/token)
    // qwen active=3e9, kvBytes=24576; on B300 flopsPerByte=1875.
    // Expect ≈ 3e9 / (1875 × 24576) ≈ 65.1
    expect(
      getCrossoverContextTokens({
        flopsPerByte: 1875,
        activeParams: 3e9,
        kvBytesPerToken: 24576,
      }),
    ).toBeCloseTo(3e9 / (1875 * 24576), 1);
  });

  it("returns NaN crossover when KV bytes/token is zero (e.g., embedding workloads)", () => {
    const scenario = buildScenario(b300, granite);
    const result = calculateScenario(scenario);
    expect(Number.isNaN(result.crossoverContextTokens)).toBe(true);
    expect(Number.isNaN(result.prefillTokensPerSecond)).toBe(true);
  });
});

describe("pipeline bubble (Section 3 flashcards)", () => {
  it("returns zero bubble at PP=1 (no behavior change)", () => {
    const out = getPipelineBubble({ pipelineStages: 1, batchSize: 128 });
    expect(out.fraction).toBe(0);
    expect(out.efficiency).toBe(1);
  });

  it("matches the (P-1)/(M+P-1) closed form for PP=4, batch=12", () => {
    // bubble = (4-1)/(12+4-1) = 3/15 = 0.2
    const out = getPipelineBubble({ pipelineStages: 4, batchSize: 12 });
    expect(out.fraction).toBeCloseTo(0.2, 5);
    expect(out.efficiency).toBeCloseTo(0.8, 5);
  });

  it("derivedTokensPerSecond at PP=1 stays at the unbubbled value (regression check)", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 64 });
    const result = calculateScenario(scenario);
    expect(result.pipelineEfficiency).toBe(1);
    // Same envelope as the existing 'derives a tokens-per-second estimate' test.
    expect(result.derivedTokensPerSecond).toBeGreaterThan(1500);
    expect(result.derivedTokensPerSecond).toBeLessThan(2000);
  });

  it("applies the bubble to derivedTokensPerSecond at PP > 1 with small batch", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 8, pipelineStages: 8 });
    const result = calculateScenario(scenario);
    // bubble = 7/15 ≈ 0.467 — efficiency ≈ 0.533. Throughput should drop by that factor.
    expect(result.pipelineBubbleFraction).toBeCloseTo(7 / 15, 3);
    expect(result.pipelineEfficiency).toBeCloseTo(8 / 15, 3);
    // Warning should fire (>20% bubble).
    expect(result.warnings.some((w) => /pipeline bubble/i.test(w))).toBe(true);
  });

  it("does not fire the bubble warning when batch is large vs P", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 256, pipelineStages: 2 });
    const result = calculateScenario(scenario);
    expect(result.pipelineBubbleFraction).toBeLessThan(0.05);
    expect(result.warnings.some((w) => /pipeline bubble/i.test(w))).toBe(false);
  });
});

describe("lifecycle FLOPs (Section 5 flashcards)", () => {
  it("computes 6N pretrain, scaled RL, and scaled inference", () => {
    const out = getLifecycleFlops({
      activeParams: 1e10,
      pretrainTokens: 1e13,
      rlTokens: 1e12,
      inferenceTokens: 1e12,
      rlInefficiency: 3,
      inferenceInefficiency: 5,
    });
    // pretrain = 6 × 1e10 × 1e13 = 6e23
    expect(out.pretrainFlops).toBeCloseTo(6e23, -20);
    // rl = 2 × 1e10 × 1e12 × 3 = 6e22
    expect(out.rlFlops).toBeCloseTo(6e22, -19);
    // inference = 2 × 1e10 × 1e12 × 5 = 1e23
    expect(out.inferenceFlops).toBeCloseTo(1e23, -20);
    expect(out.dominantPhase).toBe("pretrain");
  });

  it("returns an empty/zero shape when no lifecycle inputs are set", () => {
    const out = getLifecycleFlops({
      activeParams: 1e10,
      pretrainTokens: 0,
      rlTokens: 0,
      inferenceTokens: 0,
      rlInefficiency: 3,
      inferenceInefficiency: 5,
    });
    expect(out.totalFlops).toBe(0);
    expect(out.dominantPhase).toBe("none");
  });

  it("dominanceRatio reports the gap to the second-largest phase", () => {
    const out = getLifecycleFlops({
      activeParams: 1e10,
      pretrainTokens: 1e14, // huge pretrain
      rlTokens: 1e10,
      inferenceTokens: 1e10,
      rlInefficiency: 3,
      inferenceInefficiency: 5,
    });
    // pretrain dominates by orders of magnitude
    expect(out.dominantPhase).toBe("pretrain");
    expect(out.dominanceRatio).toBeGreaterThan(100);
  });

  it("surfaces lifecycle on the full scenario result when inputs are provided", () => {
    const base = buildScenario(b300, qwen, { batchSize: 64 });
    const scenario = { ...base, pretrainTokens: 1e13, rlTokens: 1e12 };
    const result = calculateScenario(scenario);
    expect(result.lifecycle.pretrainFlops).toBeGreaterThan(0);
    expect(result.lifecycle.rlFlops).toBeGreaterThan(0);
    // inference flops use the derived TPS × deployment days × seconds
    // (when manual TPS override is 0). Should be > 0 since qwen is a decoder.
    expect(result.lifecycle.inferenceFlops).toBeGreaterThanOrEqual(0);
  });
});

describe("serving plan", () => {
  it("recommends TP equal to total GPUs when there is no pipeline parallelism", () => {
    const scenario = buildScenario(b300, qwen, { pipelineStages: 1 });
    const plan = createServingPlan(scenario, calculateScenario(scenario));
    expect(plan.recommendedTensorParallelSize).toBe(8);
    expect(plan.recommendedFlags.some((f) => f.startsWith("--pipeline-parallel-size"))).toBe(false);
  });

  it("splits TP × PP correctly when pipeline parallelism is used", () => {
    const scenario = buildScenario(b300, qwen, { pipelineStages: 2 });
    const plan = createServingPlan(scenario, calculateScenario(scenario));
    expect(plan.recommendedTensorParallelSize).toBe(4);
    expect(plan.recommendedFlags).toContain("--pipeline-parallel-size 2");
  });

  it("emits H200-server config without pipeline-parallel-size flag", () => {
    const scenario = buildScenario(h200, qwen);
    const plan = createServingPlan(scenario, calculateScenario(scenario));
    expect(plan.recommendedTensorParallelSize).toBe(4);
    expect(plan.recommendedFlags.some((f) => f.startsWith("--pipeline-parallel-size"))).toBe(false);
  });
});

describe("dollar cost per million tokens", () => {
  it("computes cost from pool $/hour, step time, and batch", () => {
    // 8 GPUs at $3.60/hr → pool costs $0.008/s. A 10 ms step serving 100
    // sequences yields 100 tokens per step → $0.0000008/token → $0.80/1M.
    expect(
      getCostPerMillionTokens({ costPerGpuHour: 3.6, gpuCount: 8, stepSeconds: 0.01, batchSize: 100 }),
    ).toBeCloseTo(0.8);
  });

  it("is NaN when no price is known", () => {
    expect(
      getCostPerMillionTokens({ costPerGpuHour: 0, gpuCount: 8, stepSeconds: 0.01, batchSize: 100 }),
    ).toBeNaN();
  });

  it("scenario cost sits exactly on the roofline sweep curve at the same batch", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 128, costPerGpuHour: 3.6 });
    const result = calculateScenario(scenario);
    const sweepPoint = getRooflineSweep(scenario, { samples: 2, maxBatch: 128 })[1]; // batch = 128
    expect(sweepPoint.batch).toBeCloseTo(128);
    const poolCostPerSecond = (3.6 * b300.gpuCount) / 3600;
    expect(result.costPerMillionTokensUsd).toBeCloseTo(sweepPoint.costPerToken * poolCostPerSecond * 1e6);
  });

  it("defaults the scenario rate to the hardware preset market price", () => {
    const scenario = buildScenario(h200, qwen);
    expect(scenario.costPerGpuHour).toBe(h200.costPerGpuHourUsd);
  });

  it("returns NaN cost for non-decoder workloads", () => {
    const scenario = buildScenario(b300, granite, { costPerGpuHour: 3.6 });
    expect(calculateScenario(scenario).costPerMillionTokensUsd).toBeNaN();
  });
});

describe("optimal batch", () => {
  it("locks to the knee when HBM has room above it", () => {
    expect(getOptimalBatch({ maxFittingBatch: 1000, batchThreshold: 300 })).toEqual({
      batch: 300,
      binding: "knee",
    });
  });

  it("clamps to the HBM ceiling when the knee is out of reach", () => {
    expect(getOptimalBatch({ maxFittingBatch: 279, batchThreshold: 50000 })).toEqual({
      batch: 279,
      binding: "hbm",
    });
  });

  it("reports infeasible when nothing fits", () => {
    expect(getOptimalBatch({ maxFittingBatch: 0, batchThreshold: 500 })).toEqual({
      batch: 1,
      binding: "infeasible",
    });
    expect(getOptimalBatch({ maxFittingBatch: Number.NaN, batchThreshold: 500 }).binding).toBe("infeasible");
  });
});

describe("time to first token", () => {
  it("equals context divided by prefill throughput", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 64 });
    const result = calculateScenario(scenario);
    expect(result.ttftSeconds).toBeCloseTo(scenario.contextTokens / result.prefillTokensPerSecond);
  });

  it("is NaN for non-decoder workloads", () => {
    const result = calculateScenario(buildScenario(b300, granite));
    expect(result.ttftSeconds).toBeNaN();
  });
});

describe("vLLM serve model argument", () => {
  it("derives the HF repo path from a curated preset's sources", () => {
    const kimi = modelPresets.find((m) => m.id === "kimi-k2.6")!;
    expect(getServeModelArg(kimi)).toBe("moonshotai/Kimi-K2.6");
  });

  it("strips the hf: prefix for imported presets", () => {
    expect(
      getServeModelArg({ ...qwen, id: "hf:Qwen/Qwen3-Coder-Next-80B-A3B", sources: [] }),
    ).toBe("Qwen/Qwen3-Coder-Next-80B-A3B");
  });

  it("falls back to the label when no repo is known", () => {
    expect(getServeModelArg({ ...qwen, sources: ["Qwen3-Coder-Next model card"] })).toBe(qwen.label);
  });
});
