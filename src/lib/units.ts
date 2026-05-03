export function formatNumber(value: number, decimals = 1): string {
  if (!Number.isFinite(value)) return "--";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(decimals);
  if (abs >= 1) return value.toFixed(Math.min(2, decimals + 1));
  return value.toPrecision(2);
}

export function formatCompact(value: number, suffix = ""): string {
  if (!Number.isFinite(value)) return "--";
  const abs = Math.abs(value);
  const units: Array<[number, string]> = [
    [1e15, "P"],
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  const match = units.find(([scale]) => abs >= scale);
  if (!match) return `${formatNumber(value)}${suffix}`;
  return `${formatNumber(value / match[0])}${match[1]}${suffix}`;
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value)) return "--";
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${formatNumber(value / 1e12)} TB`;
  if (abs >= 1e9) return `${formatNumber(value / 1e9)} GB`;
  if (abs >= 1e6) return `${formatNumber(value / 1e6)} MB`;
  if (abs >= 1e3) return `${formatNumber(value / 1e3)} KB`;
  return `${formatNumber(value)} B`;
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "--";
  if (seconds >= 1) return `${formatNumber(seconds)} s`;
  if (seconds >= 1e-3) return `${formatNumber(seconds * 1e3)} ms`;
  if (seconds >= 1e-6) return `${formatNumber(seconds * 1e6)} us`;
  return `${formatNumber(seconds * 1e9)} ns`;
}

