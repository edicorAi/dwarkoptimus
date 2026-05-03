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

export async function searchModels(query: string, options: HfRequestOptions = {}): Promise<HfSearchHit[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];
  // `filter=text-generation` keeps the noise low; `sort=downloads` ranks the popular ones first.
  const url = `${HF_API}/models?search=${encodeURIComponent(trimmed)}&filter=text-generation&sort=downloads&direction=-1&limit=15`;
  const res = await fetch(url, { signal: options.signal, headers: authHeaders(options.token) });
  if (!res.ok) throw new Error(`Hugging Face search failed (${res.status})`);
  const data = (await res.json()) as HfSearchHit[];
  return data;
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
  num_experts_per_tok?: number;
  moe_intermediate_size?: number;
  // safetensors metadata (sometimes present on the model card response)
  safetensors?: { total?: number; parameters?: Record<string, number> };
};

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
  return (await res.json()) as HfModelConfig;
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

// KV bytes per token from architecture: 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes.
// vLLM serves KV at the model's torch_dtype unless the operator overrides with --kv-cache-dtype,
// so this matches the "out-of-the-box" estimate.
export function deriveKvBytesPerToken(config: HfModelConfig): number {
  const layers = config.num_hidden_layers ?? 0;
  const kvHeads = config.num_key_value_heads ?? config.num_attention_heads ?? 0;
  const headDim =
    config.head_dim ??
    (config.hidden_size && config.num_attention_heads ? config.hidden_size / config.num_attention_heads : 0);
  const dtypeBytes = dtypeToBytes(config.torch_dtype);
  if (!layers || !kvHeads || !headDim) return 0;
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
  const numExperts = config.num_local_experts ?? config.num_experts ?? 1;
  const ffnPerLayer = numExperts > 1
    ? numExperts * 3 * hidden * (config.moe_intermediate_size ?? ffnDense)
    : 3 * hidden * ffnDense;
  const layerParams = layers * (attnPerLayer + ffnPerLayer);
  const vocab = config.vocab_size ?? 32000;
  const embedding = vocab * hidden * (config.tie_word_embeddings ? 1 : 2);
  return layerParams + embedding;
}

export function estimateActiveParams(config: HfModelConfig, totalParams: number): number {
  const numExperts = config.num_local_experts ?? config.num_experts ?? 1;
  const activated = config.num_experts_per_tok ?? 0;
  if (numExperts <= 1 || activated <= 0) return totalParams;
  // Active fraction ≈ attention (always on) + ffn × activated/total experts.
  // For a typical MoE the FFN dominates total params; approximate active ≈ total × activated/numExperts
  // plus the attention portion, which we don't break out separately. Slight over-estimate.
  const moeActive = totalParams * (activated / numExperts);
  // Add a small floor for the always-on attention/embedding portion (~10%).
  return Math.max(moeActive, totalParams * 0.1 * (activated / numExperts) + moeActive);
}

export function derivePresetFromHfConfig(
  repoId: string,
  config: HfModelConfig,
  knownTotalParams?: number,
): ModelPreset {
  const dtypeBytes = dtypeToBytes(config.torch_dtype);
  const kvBytesPerToken = deriveKvBytesPerToken(config);
  const numExperts = config.num_local_experts ?? config.num_experts ?? 1;
  const isMoE = numExperts > 1;
  const architecture = isMoE ? "moe" : "dense";
  const totalParams = knownTotalParams ?? estimateTotalParams(config);
  const activeParams = estimateActiveParams(config, totalParams);
  const defaultBytesPerParam: 0.5 | 1 | 2 = dtypeBytes <= 0.5 ? 0.5 : dtypeBytes <= 1 ? 1 : 2;
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
    kvConfidence: kvBytesPerToken > 0 ? "source-backed" : "unknown",
    notes: `Imported from huggingface.co/${repoId}. KV size derived from layers × kv_heads × head_dim × dtype.${
      knownTotalParams ? "" : " Total params estimated from architecture; verify before production."
    }`,
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
