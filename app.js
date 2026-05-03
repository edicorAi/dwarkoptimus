const defaults = {
  hardwarePreset: "custom",
  modelPreset: "custom",
  totalParams: "700",
  totalParamsUnit: "1000000000",
  activeParams: "100",
  activeParamsUnit: "1000000000",
  contextLength: "200",
  contextLengthUnit: "1000",
  kvBytes: "1700",
  weightBytes: "0.5",
  flopsPerByte: "300",
  memoryCapacity: "288",
  memoryCapacityUnit: "1000000000",
  memoryBandwidth: "20",
  memoryBandwidthUnit: "1000000000000",
  tokensPerSecond: "50",
  tokensPerSecondUnit: "1000000",
  deploymentDays: "60",
  deploymentUnit: "1",
  pipelineStages: "4",
  expertParallelism: "64",
};

const hardwarePresets = {
  b300Dell8Gpu: {
    label: "Dell B300 server (8x GPU)",
    memoryCapacity: 2304,
    memoryCapacityUnit: "1000000000",
    memoryBandwidth: 64,
    memoryBandwidthUnit: "1000000000000",
    flopsPerByte: 1875,
    note: "Your Dell B300 server treated as 8 Blackwell Ultra GPUs: about 2.3 TB HBM and 64 TB/s aggregate bandwidth, derived from GB300/B300 per-GPU NVL72 figures.",
  },
  h200Pool16Gpu: {
    label: "H200 pool (4 servers, 16x GPU)",
    memoryCapacity: 2256,
    memoryCapacityUnit: "1000000000",
    memoryBandwidth: 76.8,
    memoryBandwidthUnit: "1000000000000",
    flopsPerByte: 412,
    note: "Your four H200 servers treated as a 16-GPU aggregate pool: 16 x 141 GB HBM and 16 x 4.8 TB/s. Cross-server networking may limit real all-to-all/pipeline performance.",
  },
  h200Server4Gpu: {
    label: "H200 server (4x GPU)",
    memoryCapacity: 564,
    memoryCapacityUnit: "1000000000",
    memoryBandwidth: 19.2,
    memoryBandwidthUnit: "1000000000000",
    flopsPerByte: 412,
    note: "One of your H200 servers: 4 x 141 GB HBM and 4 x 4.8 TB/s aggregate bandwidth.",
  },
  h200Sxm: {
    label: "NVIDIA H200 SXM",
    memoryCapacity: 141,
    memoryCapacityUnit: "1000000000",
    memoryBandwidth: 4.8,
    memoryBandwidthUnit: "1000000000000",
    flopsPerByte: 412,
    note: "141 GB HBM3e, 4.8 TB/s, dense FP8 throughput ratio estimated from NVIDIA H200 specs.",
  },
  h200Nvl: {
    label: "NVIDIA H200 NVL",
    memoryCapacity: 141,
    memoryCapacityUnit: "1000000000",
    memoryBandwidth: 4.8,
    memoryBandwidthUnit: "1000000000000",
    flopsPerByte: 348,
    note: "141 GB HBM3e, 4.8 TB/s, dense FP8 throughput ratio estimated from NVIDIA H200 NVL specs.",
  },
  gb300B300: {
    label: "NVIDIA B300 / GB300 NVL72 per GPU",
    memoryCapacity: 288,
    memoryCapacityUnit: "1000000000",
    memoryBandwidth: 8,
    memoryBandwidthUnit: "1000000000000",
    flopsPerByte: 1875,
    note: "Per-GPU values derived from GB300 NVL72 rack totals: about 20 TB GPU memory and 576 TB/s across 72 GPUs; dense FP4 ratio uses 1080 PFLOPS rack throughput.",
  },
};

const modelPresets = {
  qwen3CoderNext: {
    label: "qwen3-coder-next",
    totalParams: 80,
    activeParams: 3,
    contextLength: 262.144,
    kvBytes: 24576,
    note: "From LiteLLM: qwen3-coder-next. 80B total, 3B active, 262K context. KV estimate uses the 12 gated-attention layers, 2 KV heads, 256 head dim, BF16.",
  },
  kimiK26: {
    label: "kimi-k2.6",
    totalParams: 1000,
    activeParams: 32,
    contextLength: 256,
    kvBytes: 70000,
    note: "From LiteLLM: kimi-k2.6. Public docs emphasize K2.6 capabilities but not all architecture fields; using Kimi K2-family 1T total, 32B active, 256K context and a rough MLA-cache estimate.",
  },
  qwen35: {
    label: "qwen3.5",
    totalParams: 122,
    activeParams: 10,
    contextLength: 262.144,
    kvBytes: 32768,
    note: "From LiteLLM: qwen3.5:cloud. Mapped to Qwen3.5-122B-A10B: 122B total, 10B active, 262K native context.",
  },
  qwen35_397b: {
    label: "qwen3.5-397b",
    totalParams: 397,
    activeParams: 17,
    contextLength: 262.144,
    kvBytes: 30720,
    note: "From LiteLLM: qwen3.5:397b-cloud. 397B total, 17B active, 262K native context, extendable to about 1M with YaRN.",
  },
  glm51: {
    label: "glm-5.1",
    totalParams: 744,
    activeParams: 40,
    contextLength: 200,
    kvBytes: 1916928,
    note: "From LiteLLM: glm-5.1. Public specs vary between 744B and 754B total; using 744B total, 40B active, 200K context. KV estimate is an upper-style BF16 cache estimate; GLM-5.1 uses sparse/latent attention, so adjust for the actual runtime.",
  },
  gemma4_31b: {
    label: "gemma4-31b",
    totalParams: 31,
    activeParams: 31,
    contextLength: 256,
    kvBytes: 1700,
    note: "From LiteLLM: gemma4:31b-cloud. 31B dense, 256K context. KV bytes left at the lecture crossover default; tune if you know Gemma 4 serving cache details.",
  },
  mistralLarge3: {
    label: "mistral-large-3",
    totalParams: 675,
    activeParams: 41,
    contextLength: 256,
    kvBytes: 1700,
    note: "From LiteLLM: mistral-large-3:675b-cloud. 675B total, 41B active, 256K context. KV bytes is a placeholder until deployment cache details are known.",
  },
  minimaxM25: {
    label: "minimax-m2.5",
    totalParams: 229,
    activeParams: 10,
    contextLength: 1048.576,
    kvBytes: 1700,
    note: "From LiteLLM: minimax-m2.5. 229B total, 10B active, 1M context. Uses hybrid Lightning/SoftMax attention, so KV behavior is not the same as dense attention.",
  },
  nemotron3Super: {
    label: "nemotron-3-super",
    totalParams: 120,
    activeParams: 12,
    contextLength: 1000,
    kvBytes: 1700,
    note: "From LiteLLM: nemotron-3-super. 120B total, 12B active, up to 1M context. Hybrid Mamba-Transformer/LatentMoE means KV estimates are approximate.",
  },
  ministral3_14b: {
    label: "ministral-3-14b",
    totalParams: 14,
    activeParams: 14,
    contextLength: 256,
    kvBytes: 1700,
    note: "From LiteLLM: ministral-3:14b-cloud. 14B dense, 256K context.",
  },
  kimiK25: {
    label: "kimi-k2.5",
    totalParams: 1000,
    activeParams: 32,
    contextLength: 256,
    kvBytes: 70000,
    note: "From LiteLLM: kimi-k2.5. 1T total, 32B active, 256K context, Kimi K2-family MLA cache estimate.",
  },
  gptOss120: {
    label: "gpt-oss-120b",
    totalParams: 117,
    activeParams: 5.1,
    contextLength: 128,
    kvBytes: 65536,
    note: "From LiteLLM: gpt-oss:120b-cloud. 117B total, 5.1B active, 128K context.",
  },
  gptOss20: {
    label: "gpt-oss-20b",
    totalParams: 21,
    activeParams: 3.6,
    contextLength: 128,
    kvBytes: 32768,
    note: "From LiteLLM: gpt-oss:20b-cloud. 21B total, 3.6B active, 128K context.",
  },
  deepseekV4Flash: {
    label: "deepseek-v4-flash",
    totalParams: 284,
    activeParams: 13,
    contextLength: 1000,
    kvBytes: 70000,
    note: "From LiteLLM: deepseek-v4-flash. 284B total, 13B active, 1M context. KV estimate is rough for DeepSeek-style efficient attention.",
  },
  graniteEmbedding107m: {
    label: "granite-embedding-107m",
    totalParams: 0.107,
    activeParams: 0.107,
    contextLength: 0.512,
    kvBytes: 0,
    note: "From LiteLLM: IBM Granite Embedding 107M Multilingual. Encoder embedding model, 107M params, 512 token max sequence, 384-dim output. Decode/KV calculations do not apply.",
  },
  bgeM3: {
    label: "bge-m3",
    totalParams: 0.569,
    activeParams: 0.569,
    contextLength: 8.192,
    kvBytes: 0,
    note: "From LiteLLM: BAAI BGE-M3. Encoder embedding model, 569M params, 8192 token max sequence, 1024-dim dense output. Decode/KV calculations do not apply.",
  },
  graniteDocling258m: {
    label: "granite-docling-258m",
    totalParams: 0.258,
    activeParams: 0.258,
    contextLength: 8.192,
    kvBytes: 0,
    note: "From LiteLLM: IBM Granite Docling 258M. Compact document VLM, not a general decoder LLM; use these fields only as a rough size placeholder.",
  },
};

const ids = Object.keys(defaults);
const $ = (id) => document.getElementById(id);

function initPresets() {
  fillPresetSelect("hardwarePreset", hardwarePresets);
  fillPresetSelect("modelPreset", modelPresets);
}

function fillPresetSelect(id, presets) {
  const select = $(id);
  Object.entries(presets).forEach(([value, preset]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = preset.label;
    select.append(option);
  });
}

function readNumber(id) {
  const value = Number($(id).value);
  return Number.isFinite(value) ? value : 0;
}

function readScaled(valueId, unitId) {
  return readNumber(valueId) * readNumber(unitId);
}

function fmtCompact(value, unit = "") {
  if (!Number.isFinite(value)) return "--";
  const abs = Math.abs(value);
  const suffixes = [
    [1e15, "P"],
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  const match = suffixes.find(([scale]) => abs >= scale);
  if (!match) return `${round(value)}${unit}`;
  return `${round(value / match[0])}${match[1]}${unit}`;
}

function fmtBytes(value) {
  if (!Number.isFinite(value)) return "--";
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${round(value / 1e12)} TB`;
  if (abs >= 1e9) return `${round(value / 1e9)} GB`;
  if (abs >= 1e6) return `${round(value / 1e6)} MB`;
  if (abs >= 1e3) return `${round(value / 1e3)} KB`;
  return `${round(value)} B`;
}

function fmtTime(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  if (seconds >= 1) return `${round(seconds)} s`;
  if (seconds >= 1e-3) return `${round(seconds * 1e3)} ms`;
  if (seconds >= 1e-6) return `${round(seconds * 1e6)} us`;
  return `${round(seconds * 1e9)} ns`;
}

function round(value) {
  if (!Number.isFinite(value)) return "--";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toPrecision(2);
}

function getInputs() {
  const totalParams = readScaled("totalParams", "totalParamsUnit");
  const activeParams = readScaled("activeParams", "activeParamsUnit");
  const contextLength = readScaled("contextLength", "contextLengthUnit");
  const kvBytes = readNumber("kvBytes");
  const weightBytes = readNumber("weightBytes");
  const flopsPerByte = readNumber("flopsPerByte");
  const memoryCapacity = readScaled("memoryCapacity", "memoryCapacityUnit");
  const memoryBandwidth = readScaled("memoryBandwidth", "memoryBandwidthUnit");
  const tokensPerSecond = readScaled("tokensPerSecond", "tokensPerSecondUnit");
  const deploymentDays = readNumber("deploymentDays") * readNumber("deploymentUnit");
  const pipelineStages = Math.max(1, readNumber("pipelineStages"));
  const expertParallelism = Math.max(1, readNumber("expertParallelism"));

  return {
    totalParams,
    activeParams,
    contextLength,
    kvBytes,
    weightBytes,
    flopsPerByte,
    memoryCapacity,
    memoryBandwidth,
    tokensPerSecond,
    deploymentDays,
    pipelineStages,
    expertParallelism,
  };
}

function compute(values) {
  const inverseSparsity = safeDivide(values.totalParams, values.activeParams);
  const batchThreshold = values.flopsPerByte * inverseSparsity;
  const drainSeconds = safeDivide(values.memoryCapacity, values.memoryBandwidth);
  const impliedKvBytes = safeDivide(values.activeParams, values.contextLength * values.flopsPerByte);
  const inferenceTokens = values.tokensPerSecond * values.deploymentDays * 86400;
  const chinchillaTokens = 20 * values.activeParams;
  const overChinchilla = safeDivide(inferenceTokens, chinchillaTokens);
  const localBatch = batchThreshold;
  const globalBatch = localBatch * values.pipelineStages;
  const weightsPerGpu = safeDivide(
    values.totalParams * values.weightBytes,
    values.expertParallelism * values.pipelineStages,
  );
  const kvPerGpu = safeDivide(
    globalBatch * values.contextLength * values.kvBytes,
    values.expertParallelism * values.pipelineStages,
  );

  return {
    inverseSparsity,
    batchThreshold,
    drainSeconds,
    impliedKvBytes,
    inferenceTokens,
    chinchillaTokens,
    overChinchilla,
    localBatch,
    globalBatch,
    weightsPerGpu,
    kvPerGpu,
  };
}

function safeDivide(numerator, denominator) {
  if (!denominator) return Number.NaN;
  return numerator / denominator;
}

function update() {
  const values = getInputs();
  const result = compute(values);
  const hardware = hardwarePresets[$("hardwarePreset").value];
  const model = modelPresets[$("modelPreset").value];

  $("strip-ratio").textContent = round(values.flopsPerByte);
  $("strip-active").textContent = fmtCompact(values.activeParams);
  $("strip-context").textContent = fmtCompact(values.contextLength);

  $("batchThreshold").textContent = fmtCompact(result.batchThreshold);
  $("batchDetail").textContent =
    `inverse sparsity ${round(result.inverseSparsity)}x; local batch is ${fmtCompact(result.localBatch)} sequences`;
  $("drainTime").textContent = fmtTime(result.drainSeconds);
  $("impliedKv").textContent = fmtBytes(result.impliedKvBytes);
  $("inferenceTokens").textContent = fmtCompact(result.inferenceTokens, " tokens");
  $("overChinchilla").textContent = `${round(result.overChinchilla)}x`;

  $("weightsMemory").textContent = fmtBytes(result.weightsPerGpu);
  $("kvMemory").textContent = fmtBytes(result.kvPerGpu);
  const maxMemory = Math.max(result.weightsPerGpu, result.kvPerGpu, 1);
  $("weightsBar").style.width = `${Math.max(3, (result.weightsPerGpu / maxMemory) * 100)}%`;
  $("kvBar").style.width = `${Math.max(3, (result.kvPerGpu / maxMemory) * 100)}%`;
  $("pipelineNote").textContent =
    `Uses ${round(values.pipelineStages)} stages and ${round(values.expertParallelism)}-way expert parallelism. ` +
    `KV stays roughly constant per GPU as stages increase because the global batch also rises.`;
  $("presetNote").textContent = [
    hardware ? `Hardware: ${hardware.note}` : "Hardware: custom values.",
    model ? `Model: ${model.note}` : "Model: custom values.",
  ].join(" ");

  drawChart(values, result);
}

function drawChart(values, result) {
  const canvas = $("latencyChart");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 58, right: 18, top: 22, bottom: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxBatch = Math.max(1, result.batchThreshold * 2.2);
  const memBw = values.memoryBandwidth || 1;
  const computeThroughput = values.flopsPerByte * memBw;

  const points = Array.from({ length: 80 }, (_, index) => {
    const batch = 1 + (index / 79) * maxBatch;
    const weight = (values.totalParams * values.weightBytes) / memBw;
    const kv = (batch * values.contextLength * values.kvBytes) / memBw;
    const computeTime = (batch * values.activeParams) / computeThroughput;
    const memoryTime = weight + kv;
    return { batch, weight, kv, computeTime, memoryTime, total: Math.max(computeTime, memoryTime) };
  });

  const maxY = Math.max(...points.flatMap((p) => [p.computeTime, p.memoryTime, p.total]), 1e-12);

  ctx.strokeStyle = "#d9dfda";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();

  drawLine(ctx, points, "computeTime", "#1f7a55", 2, pad, plotW, plotH, maxBatch, maxY);
  drawLine(ctx, points, "memoryTime", "#a66d10", 2, pad, plotW, plotH, maxBatch, maxY);
  drawLine(ctx, points, "total", "#17211d", 4, pad, plotW, plotH, maxBatch, maxY);

  const thresholdX = pad.left + (Math.min(result.batchThreshold, maxBatch) / maxBatch) * plotW;
  ctx.strokeStyle = "rgba(169, 65, 50, 0.7)";
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(thresholdX, pad.top);
  ctx.lineTo(thresholdX, height - pad.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#64706b";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillText("batch", width - 58, height - 14);
  ctx.save();
  ctx.translate(18, 88);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("time per forward pass", 0, 0);
  ctx.restore();

  drawLegend(ctx, [
    ["compute", "#1f7a55"],
    ["memory", "#a66d10"],
    ["total", "#17211d"],
    ["threshold", "#a94132"],
  ]);

  $("chartCaption").textContent = `0 to ${fmtCompact(maxBatch)} batch`;
}

function drawLine(ctx, points, key, color, lineWidth, pad, plotW, plotH, maxBatch, maxY) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = pad.left + (point.batch / maxBatch) * plotW;
    const y = pad.top + plotH - (point[key] / maxY) * plotH;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawLegend(ctx, entries) {
  ctx.font = "12px Inter, system-ui, sans-serif";
  entries.forEach(([label, color], index) => {
    const x = 76 + index * 112;
    ctx.fillStyle = color;
    ctx.fillRect(x, 18, 16, 4);
    ctx.fillStyle = "#64706b";
    ctx.fillText(label, x + 22, 22);
  });
}

function resetDefaults() {
  ids.forEach((id) => {
    $(id).value = defaults[id];
  });
  update();
}

function applyHardwarePreset() {
  const preset = hardwarePresets[$("hardwarePreset").value];
  if (!preset) {
    update();
    return;
  }
  $("memoryCapacity").value = preset.memoryCapacity;
  $("memoryCapacityUnit").value = preset.memoryCapacityUnit;
  $("memoryBandwidth").value = preset.memoryBandwidth;
  $("memoryBandwidthUnit").value = preset.memoryBandwidthUnit;
  $("flopsPerByte").value = preset.flopsPerByte;
  update();
}

function applyModelPreset() {
  const preset = modelPresets[$("modelPreset").value];
  if (!preset) {
    update();
    return;
  }
  $("totalParams").value = preset.totalParams;
  $("totalParamsUnit").value = "1000000000";
  $("activeParams").value = preset.activeParams;
  $("activeParamsUnit").value = "1000000000";
  $("contextLength").value = preset.contextLength;
  $("contextLengthUnit").value = "1000";
  $("kvBytes").value = preset.kvBytes;
  update();
}

ids.forEach((id) => {
  $(id).addEventListener("input", update);
  $(id).addEventListener("change", update);
});

$("hardwarePreset").addEventListener("change", applyHardwarePreset);
$("modelPreset").addEventListener("change", applyModelPreset);
$("reset-button").addEventListener("click", resetDefaults);
initPresets();
update();
