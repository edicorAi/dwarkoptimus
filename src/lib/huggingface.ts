import type { ModelPreset } from "../types";

const HF_API = "https://huggingface.co/api";
const HF_RESOLVE = "https://huggingface.co";

export type HfSearchHit = {
  id: string;
  modelId?: string;
  downloads?: number;
  likes?: number;
  pipeline_tag?: string;
  tags?: string[];
  // HF returns: false for public, "auto" or "manual" for gated repos requiring license acceptance.
  gated?: false | "auto" | "manual";
};

function authHeaders(token?: string): HeadersInit | undefined {
  return token ? { Authorization: `Bearer ${token}` } : undefined;
}

export type HfRequestOptions = { signal?: AbortSignal; token?: string };

// Current-generation flagships (Qwen3.5, MiniMax M3, Kimi K2.7) are natively
// multimodal and tagged `image-text-to-text` on the Hub, so filtering on
// text-generation alone hides exactly the models people search for. The API
// only accepts one pipeline filter per request, so we fan out and merge.
const SEARCH_PIPELINES = ["text-generation", "image-text-to-text"] as const;

export async function searchModels(query: string, options: HfRequestOptions = {}): Promise<HfSearchHit[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];
  const settled = await Promise.allSettled(
    SEARCH_PIPELINES.map(async (pipeline) => {
      const url = `${HF_API}/models?search=${encodeURIComponent(trimmed)}&filter=${pipeline}&sort=downloads&direction=-1&limit=15`;
      const res = await fetch(url, { signal: options.signal, headers: authHeaders(options.token) });
      if (!res.ok) throw new Error(`Hugging Face search failed (${res.status})`);
      return (await res.json()) as HfSearchHit[];
    }),
  );
  const fulfilled = settled.filter((entry): entry is PromiseFulfilledResult<HfSearchHit[]> => entry.status === "fulfilled");
  // Only fail the search when every pipeline failed (e.g. offline, aborted).
  if (fulfilled.length === 0) throw (settled[0] as PromiseRejectedResult).reason;
  const byId = new Map<string, HfSearchHit>();
  for (const entry of fulfilled) {
    for (const hit of entry.value) {
      const id = hit.id ?? hit.modelId;
      if (id && !byId.has(id)) byId.set(id, hit);
    }
  }
  return [...byId.values()].sort((a, b) => (b.downloads ?? 0) - (a.downloads ?? 0)).slice(0, 15);
}

export type HfModelConfig = {
  model_type?: string;
  architectures?: string[];
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  torch_dtype?: string;
  tie_word_embeddings?: boolean;
  intermediate_size?: number;
  // MoE: different vendors use different keys; we read whatever exists.
  num_local_experts?: number;
  num_experts?: number;
  n_routed_experts?: number;
  num_experts_per_tok?: number;
  moe_intermediate_size?: number;
  // MLA (DeepSeek / GLM / Kimi): the KV cache holds one compressed latent per
  // token per layer instead of full K/V heads.
  kv_lora_rank?: number;
  qk_rope_head_dim?: number;
  // DSA-style sparse-attention indexer (DeepSeek V4, GLM 5.x): adds a small
  // per-layer index cache our formula does not count.
  index_topk?: number;
  // Multimodal wrappers (Qwen3-VL, MiniMax M3, Kimi K2.7) nest the decoder
  // config; we flatten it in normalizeHfConfig before deriving anything.
  text_config?: HfModelConfig;
  language_config?: HfModelConfig;
  quantization_config?: { quant_method?: string; quant_algo?: string; bits?: number };
  // safetensors metadata (sometimes present on the model card response)
  safetensors?: { total?: number; parameters?: Record<string, number> };
};

// Multimodal repos wrap the decoder under text_config/language_config
// (vision_config sits alongside it). Flatten so the transformer fields win,
// while top-level-only fields (torch_dtype, quantization_config) survive.
export function normalizeHfConfig(raw: HfModelConfig): HfModelConfig {
  const nested = raw.text_config ?? raw.language_config;
  if (!nested) return raw;
  return { ...raw, ...nested };
}

export async function loadModelConfig(repoId: string, options: HfRequestOptions = {}): Promise<HfModelConfig> {
  const url = `${HF_RESOLVE}/${encodeRepoId(repoId)}/resolve/main/config.json`;
  const res = await fetch(url, { signal: options.signal, headers: authHeaders(options.token) });
  if (res.status === 401 || res.status === 403) {
    throw new Error(
      options.token
        ? "Token rejected by Hugging Face. Accept the license on the model page, then try again."
        : "This model is gated. Add your Hugging Face access token below (and accept the license on the model page).",
    );
  }
  if (res.status === 404) {
    throw new Error("No config.json on this repo's main branch.");
  }
  if (!res.ok) throw new Error(`Failed to load config.json (${res.status})`);
  return normalizeHfConfig((await res.json()) as HfModelConfig);
}

// Parameter count published on the model page (preferred over our estimator).
export async function loadSafetensorsTotal(repoId: string, options: HfRequestOptions = {}): Promise<number | undefined> {
  const url = `${HF_API}/models/${encodeRepoId(repoId)}`;
  const res = await fetch(url, { signal: options.signal, headers: authHeaders(options.token) });
  if (!res.ok) return undefined;
  const data = (await res.json()) as HfModelConfig;
  return data.safetensors?.total;
}

const TOKEN_KEY = "dwarkoptimus.hf-token";

export function loadStoredHfToken(): string {
  if (typeof localStorage === "undefined") return "";
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function storeHfToken(token: string): void {
  if (typeof localStorage === "undefined") return;
  if (token.trim()) localStorage.setItem(TOKEN_KEY, token.trim());
  else localStorage.removeItem(TOKEN_KEY);
}

export function dtypeToBytes(dtype?: string): number {
  if (!dtype) return 2;
  const d = dtype.toLowerCase();
  if (d.includes("fp4") || d.includes("int4") || d.includes("nf4")) return 0.5;
  if (d.includes("fp8") || d.includes("int8")) return 1;
  if (d.includes("bf16") || d.includes("float16") || d.includes("fp16") || d.includes("half")) return 2;
  if (d.includes("float32") || d.includes("fp32")) return 4;
  return 2;
}

// KV bytes per token from architecture. Three cases, checked in this order
// (GLM-5.x configs carry BOTH kv_lora_rank and full GQA head fields — the MLA
// branch must win or the estimate is ~40× too high):
//   1. MLA (kv_lora_rank set): layers × (kv_lora_rank + qk_rope_head_dim) — one
//      compressed latent + rope key per layer, no separate K and V.
//   2. MLA in MQA-absorbed form (DeepSeek V4: one KV head whose head_dim IS the
//      latent): layers × (head_dim + qk_rope_head_dim).
//   3. GQA/MHA: 2 (K and V) × layers × kv_heads × head_dim.
// All at dtype_bytes from torch_dtype — vLLM serves KV at the model's dtype
// unless the operator overrides with --kv-cache-dtype.
export function deriveKvBytesPerToken(config: HfModelConfig): number {
  const layers = config.num_hidden_layers ?? 0;
  const dtypeBytes = dtypeToBytes(config.torch_dtype);
  if (!layers) return 0;
  if (config.kv_lora_rank) {
    return layers * (config.kv_lora_rank + (config.qk_rope_head_dim ?? 0)) * dtypeBytes;
  }
  if (config.num_key_value_heads === 1 && config.qk_rope_head_dim && config.head_dim) {
    return layers * (config.head_dim + config.qk_rope_head_dim) * dtypeBytes;
  }
  const kvHeads = config.num_key_value_heads ?? config.num_attention_heads ?? 0;
  const headDim =
    config.head_dim ??
    (config.hidden_size && config.num_attention_heads ? config.hidden_size / config.num_attention_heads : 0);
  if (!kvHeads || !headDim) return 0;
  return 2 * layers * kvHeads * headDim * dtypeBytes;
}

// Rough total param estimator from the transformer config. Used only as a fallback
// when the safetensors total isn't exposed on the model page.
export function estimateTotalParams(config: HfModelConfig): number {
  const hidden = config.hidden_size ?? 0;
  const layers = config.num_hidden_layers ?? 0;
  if (!hidden || !layers) return 0;
  const ffnDense = config.intermediate_size ?? hidden * 4;
  // Q,K,V,O projections — even when GQA is used, the K/V projections scale with kv_heads,
  // so we approximate with attention as 4×hidden² which is correct for full MHA and a
  // mild over-count for GQA. KV count matters far more for KV cache than for weight count.
  const attnPerLayer = 4 * hidden * hidden;
  const numExperts = expertCount(config);
  const ffnPerLayer = numExperts > 1
    ? numExperts * 3 * hidden * (config.moe_intermediate_size ?? ffnDense)
    : 3 * hidden * ffnDense;
  const layerParams = layers * (attnPerLayer + ffnPerLayer);
  const vocab = config.vocab_size ?? 32000;
  const embedding = vocab * hidden * (config.tie_word_embeddings ? 1 : 2);
  return layerParams + embedding;
}

// MoE expert-count key differs by vendor: Mixtral uses num_local_experts,
// Qwen uses num_experts, DeepSeek/GLM/Kimi use n_routed_experts.
function expertCount(config: HfModelConfig): number {
  return config.num_local_experts ?? config.num_experts ?? config.n_routed_experts ?? 1;
}

export function estimateActiveParams(config: HfModelConfig, totalParams: number): number {
  const numExperts = expertCount(config);
  const activated = config.num_experts_per_tok ?? 0;
  if (numExperts <= 1 || activated <= 0) return totalParams;
  // Active fraction ≈ attention (always on) + ffn × activated/total experts.
  // For a typical MoE the FFN dominates total params; approximate active ≈ total × activated/numExperts
  // plus the attention portion, which we don't break out separately. Slight over-estimate.
  const moeActive = totalParams * (activated / numExperts);
  // Add a small floor for the always-on attention/embedding portion (~10%).
  return Math.max(moeActive, totalParams * 0.1 * (activated / numExperts) + moeActive);
}

// Quantized repos (AWQ/GPTQ/NVFP4/MXFP4/FP8) keep torch_dtype at bf16, so the
// real on-disk weight width lives in quantization_config instead.
export function quantBytesPerParam(config: HfModelConfig): number | undefined {
  const quant = config.quantization_config;
  if (!quant) return undefined;
  const tag = `${quant.quant_method ?? ""} ${quant.quant_algo ?? ""}`.toLowerCase();
  if (quant.bits === 4 || /fp4|int4|nf4|awq|gptq/.test(tag)) return 0.5;
  if (quant.bits === 8 || /fp8|int8/.test(tag)) return 1;
  return undefined;
}

export function derivePresetFromHfConfig(
  repoId: string,
  config: HfModelConfig,
  knownTotalParams?: number,
): ModelPreset {
  const dtypeBytes = quantBytesPerParam(config) ?? dtypeToBytes(config.torch_dtype);
  const kvBytesPerToken = deriveKvBytesPerToken(config);
  const isMoE = expertCount(config) > 1;
  // Multimodal decoders still deserve full decode/KV math — the `vlm`
  // architecture (which skips it) is reserved for non-generative vision stacks.
  const architecture = isMoE ? "moe" : "dense";
  const totalParams = knownTotalParams ?? estimateTotalParams(config);
  const activeParams = estimateActiveParams(config, totalParams);
  const defaultBytesPerParam: 0.5 | 1 | 2 = dtypeBytes <= 0.5 ? 0.5 : dtypeBytes <= 1 ? 1 : 2;
  // DSA-style sparse attention adds a per-layer indexer cache our formula
  // skips, so those models get "estimated" rather than "source-backed".
  const hasIndexerCache = Boolean(config.index_topk);
  return {
    id: `hf:${repoId}`,
    label: repoId,
    architecture,
    totalParams,
    activeParams,
    contextTokens: config.max_position_embeddings ?? 8192,
    defaultWeightBytesPerParam: defaultBytesPerParam,
    kvBytesPerToken,
    activatedExperts: isMoE ? config.num_experts_per_tok : undefined,
    approxLayers: config.num_hidden_layers,
    confidence: knownTotalParams ? "source-backed" : "estimated",
    kvConfidence: kvBytesPerToken > 0 ? (hasIndexerCache ? "estimated" : "source-backed") : "unknown",
    notes: `Imported from huggingface.co/${repoId}. KV size derived from config.json attention shape (MLA-aware).${
      hasIndexerCache ? " Sparse-attention indexer cache not counted — treat KV as a floor." : ""
    }${knownTotalParams ? "" : " Total params estimated from architecture; verify before production."}`,
    sources: [`https://huggingface.co/${repoId}/blob/main/config.json`],
  };
}

function encodeRepoId(repoId: string): string {
  // Repo IDs are "org/name" — encode each segment, keep the slash.
  return repoId.split("/").map(encodeURIComponent).join("/");
}

const CACHE_KEY_PREFIX = "dwarkoptimus.hf-preset.";

export function loadCachedHfPresets(): ModelPreset[] {
  if (typeof localStorage === "undefined") return [];
  const out: ModelPreset[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!key || !key.startsWith(CACHE_KEY_PREFIX)) continue;
    try {
      const raw = localStorage.getItem(key);
      if (raw) out.push(JSON.parse(raw) as ModelPreset);
    } catch {
      // ignore malformed cache entries
    }
  }
  return out;
}

export function cacheHfPreset(preset: ModelPreset): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(`${CACHE_KEY_PREFIX}${preset.id}`, JSON.stringify(preset));
  } catch {
    // quota exceeded etc — just drop silently
  }
}

export function clearCachedHfPreset(presetId: string): void {
  if (typeof localStorage === "undefined") return;
  localStorage.removeItem(`${CACHE_KEY_PREFIX}${presetId}`);
}
