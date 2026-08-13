import { describe, expect, it } from "vitest";
import {
  deriveKvBytesPerToken,
  derivePresetFromHfConfig,
  dtypeToBytes,
  estimateActiveParams,
  estimateTotalParams,
  normalizeHfConfig,
  quantBytesPerParam,
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

// GLM-5.2-style MLA config: carries BOTH kv_lora_rank and full GQA head fields.
// The MLA branch must win — the GQA formula would be ~40× too high.
const glmMlaConfig: HfModelConfig = {
  model_type: "glm_moe_dsa",
  num_hidden_layers: 78,
  hidden_size: 6144,
  num_attention_heads: 64,
  num_key_value_heads: 64,
  head_dim: 192,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  n_routed_experts: 256,
  num_experts_per_tok: 8,
  moe_intermediate_size: 2048,
  intermediate_size: 12288,
  max_position_embeddings: 1048576,
  torch_dtype: "bfloat16",
  index_topk: 2048,
};

// DeepSeek-V4-style MQA-absorbed MLA: one KV head whose head_dim is the latent.
const deepseekV4Config: HfModelConfig = {
  model_type: "deepseek_v4",
  num_hidden_layers: 61,
  hidden_size: 7168,
  num_attention_heads: 128,
  num_key_value_heads: 1,
  head_dim: 512,
  qk_rope_head_dim: 64,
  n_routed_experts: 384,
  num_experts_per_tok: 6,
  max_position_embeddings: 1048576,
  torch_dtype: "bfloat16",
};

// MiniMax-M3-style multimodal wrapper: decoder nested under text_config.
const minimaxWrapperConfig: HfModelConfig = {
  model_type: "minimax_m3_vl",
  torch_dtype: "bfloat16",
  text_config: {
    hidden_size: 6144,
    num_hidden_layers: 60,
    num_attention_heads: 64,
    num_key_value_heads: 4,
    head_dim: 128,
    max_position_embeddings: 1048576,
    num_local_experts: 128,
    num_experts_per_tok: 4,
  },
};

describe("normalizeHfConfig", () => {
  it("flattens text_config while keeping top-level-only fields", () => {
    const flat = normalizeHfConfig(minimaxWrapperConfig);
    expect(flat.num_hidden_layers).toBe(60);
    expect(flat.num_key_value_heads).toBe(4);
    expect(flat.max_position_embeddings).toBe(1048576);
    expect(flat.torch_dtype).toBe("bfloat16");
  });

  it("returns the config unchanged when nothing is nested", () => {
    expect(normalizeHfConfig(llama3Config)).toBe(llama3Config);
  });
});

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

  it("prefers the MLA latent over GQA heads when kv_lora_rank is present", () => {
    // 78 × (512 + 64) × 2 = 89,856 — NOT 2 × 78 × 64 × 192 × 2 ≈ 3.8 MB
    expect(deriveKvBytesPerToken(glmMlaConfig)).toBe(89856);
  });

  it("handles DeepSeek-style MQA-absorbed MLA (1 KV head = latent)", () => {
    // 61 × (512 + 64) × 2 = 70,272
    expect(deriveKvBytesPerToken(deepseekV4Config)).toBe(70272);
  });

  it("derives GQA KV from a flattened multimodal wrapper", () => {
    // 2 × 60 × 4 × 128 × 2 = 122,880
    expect(deriveKvBytesPerToken(normalizeHfConfig(minimaxWrapperConfig))).toBe(122880);
  });
});

describe("quantBytesPerParam", () => {
  it("reads 4-bit and 8-bit quantization configs", () => {
    expect(quantBytesPerParam({ quantization_config: { quant_algo: "NVFP4" } })).toBe(0.5);
    expect(quantBytesPerParam({ quantization_config: { quant_method: "awq", bits: 4 } })).toBe(0.5);
    expect(quantBytesPerParam({ quantization_config: { quant_method: "fp8" } })).toBe(1);
  });

  it("returns undefined without a quantization config", () => {
    expect(quantBytesPerParam(llama3Config)).toBeUndefined();
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

  it("imports an n_routed_experts MoE with MLA KV and marks DSA models estimated", () => {
    const preset = derivePresetFromHfConfig("zai-org/GLM-5.2", glmMlaConfig, 753e9);
    expect(preset.architecture).toBe("moe");
    expect(preset.activatedExperts).toBe(8);
    expect(preset.kvBytesPerToken).toBe(89856);
    // index_topk present → sparse-attention indexer cache not counted
    expect(preset.kvConfidence).toBe("estimated");
    expect(preset.activeParams).toBeLessThan(preset.totalParams);
  });

  it("imports a flattened multimodal wrapper as a decoder MoE (not vlm)", () => {
    const preset = derivePresetFromHfConfig("MiniMaxAI/MiniMax-M3", normalizeHfConfig(minimaxWrapperConfig), 427e9);
    expect(preset.architecture).toBe("moe");
    expect(preset.contextTokens).toBe(1048576);
    expect(preset.kvBytesPerToken).toBe(122880);
    expect(preset.kvConfidence).toBe("source-backed");
  });
});
