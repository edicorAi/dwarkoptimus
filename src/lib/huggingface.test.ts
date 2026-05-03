import { describe, expect, it } from "vitest";
import {
  deriveKvBytesPerToken,
  derivePresetFromHfConfig,
  dtypeToBytes,
  estimateActiveParams,
  estimateTotalParams,
  type HfModelConfig,
} from "./huggingface";

// Llama-3-8B-style config (GQA: 8 KV heads, 32 attn heads, head_dim 128, 32 layers, hidden 4096).
const llama3Config: HfModelConfig = {
  model_type: "llama",
  architectures: ["LlamaForCausalLM"],
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  head_dim: 128,
  intermediate_size: 14336,
  vocab_size: 128256,
  max_position_embeddings: 8192,
  torch_dtype: "bfloat16",
  tie_word_embeddings: false,
};

// Mixtral-8x7B-style config (8 experts, 2 activated per token).
const mixtralConfig: HfModelConfig = {
  model_type: "mixtral",
  architectures: ["MixtralForCausalLM"],
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  head_dim: 128,
  intermediate_size: 14336,
  vocab_size: 32000,
  max_position_embeddings: 32768,
  torch_dtype: "bfloat16",
  tie_word_embeddings: false,
  num_local_experts: 8,
  num_experts_per_tok: 2,
  moe_intermediate_size: 14336,
};

describe("dtypeToBytes", () => {
  it("maps common dtypes correctly", () => {
    expect(dtypeToBytes("bfloat16")).toBe(2);
    expect(dtypeToBytes("float16")).toBe(2);
    expect(dtypeToBytes("float32")).toBe(4);
    expect(dtypeToBytes("fp8_e4m3")).toBe(1);
    expect(dtypeToBytes("int4")).toBe(0.5);
    expect(dtypeToBytes(undefined)).toBe(2);
  });
});

describe("deriveKvBytesPerToken", () => {
  it("uses 2 × layers × kv_heads × head_dim × dtype_bytes", () => {
    // 2 × 32 × 8 × 128 × 2 = 131,072 bytes/token = 128 KiB
    expect(deriveKvBytesPerToken(llama3Config)).toBe(131072);
  });

  it("falls back to num_attention_heads when num_key_value_heads is missing", () => {
    const config: HfModelConfig = { ...llama3Config, num_key_value_heads: undefined };
    // With 32 attn heads instead of 8: 4× more
    expect(deriveKvBytesPerToken(config)).toBe(131072 * 4);
  });

  it("returns 0 with an incomplete config", () => {
    expect(deriveKvBytesPerToken({})).toBe(0);
  });
});

describe("estimateTotalParams", () => {
  it("produces a Llama-3-8B-ish ballpark", () => {
    const total = estimateTotalParams(llama3Config);
    // Expect within 30% of 8B; the rough estimator over-counts attention slightly.
    expect(total).toBeGreaterThan(6e9);
    expect(total).toBeLessThan(11e9);
  });

  it("scales by expert count for MoE", () => {
    const total = estimateTotalParams(mixtralConfig);
    // Mixtral 8×7B is ~46.7B total. Our rough estimator overshoots; allow generous bounds.
    expect(total).toBeGreaterThan(35e9);
    expect(total).toBeLessThan(80e9);
  });
});

describe("estimateActiveParams", () => {
  it("equals total for dense models", () => {
    const total = estimateTotalParams(llama3Config);
    expect(estimateActiveParams(llama3Config, total)).toBe(total);
  });

  it("scales by activated/total experts for MoE", () => {
    const total = estimateTotalParams(mixtralConfig);
    const active = estimateActiveParams(mixtralConfig, total);
    // 2 of 8 experts → roughly 25% of total params
    expect(active).toBeGreaterThan(total * 0.2);
    expect(active).toBeLessThan(total * 0.4);
  });
});

describe("derivePresetFromHfConfig", () => {
  it("imports a Llama-style dense model with source-backed KV", () => {
    const preset = derivePresetFromHfConfig("meta-llama/Meta-Llama-3-8B", llama3Config);
    expect(preset.architecture).toBe("dense");
    expect(preset.contextTokens).toBe(8192);
    expect(preset.kvBytesPerToken).toBe(131072);
    expect(preset.kvConfidence).toBe("source-backed");
    expect(preset.id).toBe("hf:meta-llama/Meta-Llama-3-8B");
    expect(preset.defaultWeightBytesPerParam).toBe(2);
    expect(preset.approxLayers).toBe(32);
  });

  it("imports a MoE model with activatedExperts populated", () => {
    const preset = derivePresetFromHfConfig("mistralai/Mixtral-8x7B-v0.1", mixtralConfig);
    expect(preset.architecture).toBe("moe");
    expect(preset.activatedExperts).toBe(2);
    expect(preset.totalParams).toBeGreaterThan(preset.activeParams);
  });

  it("prefers a known total-params count over the estimator", () => {
    const preset = derivePresetFromHfConfig("meta-llama/Meta-Llama-3-8B", llama3Config, 8.03e9);
    expect(preset.totalParams).toBe(8.03e9);
    expect(preset.confidence).toBe("source-backed");
  });
});
