import { describe, expect, it } from "vitest";
import { hardwarePresets } from "../data/hardware";
import { modelPresets } from "../data/models";
import { buildScenario, calculateScenario, createServingPlan } from "./calculations";
import { buildScenarioReport } from "./report";

const b300 = hardwarePresets.find((h) => h.id === "dell-b300-8gpu")!;
const qwen = modelPresets.find((m) => m.id === "qwen3-coder-next")!;
const granite = modelPresets.find((m) => m.id === "granite-embedding-107m")!;

const fixedDate = new Date("2026-05-03T12:00:00.000Z");

describe("buildScenarioReport", () => {
  it("includes verdict, hardware, model, configuration, metrics, and command", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 64, contextTokens: 4096 });
    const result = calculateScenario(scenario);
    const plan = createServingPlan(scenario, result);
    const md = buildScenarioReport({ hardware: b300, model: qwen, scenario, result, plan, generatedAt: fixedDate });

    expect(md).toContain("# dwarkoptimus serving report");
    expect(md).toContain(fixedDate.toISOString());
    // Section headers
    for (const heading of [
      "## Verdict",
      "## Hardware",
      "## Model",
      "## Configuration",
      "## Memory fit",
      "## Roofline metrics",
      "## Suggested vLLM serve command",
      "## Methodology",
    ]) {
      expect(md).toContain(heading);
    }
    // The vLLM command is wrapped in a fenced bash block
    expect(md).toContain("```bash");
    expect(md).toContain("vllm serve Qwen/Qwen3-Coder-Next");
    // Hardware label flows through
    expect(md).toContain(b300.label);
    // Verdict badge emoji
    expect(md).toMatch(/🟢 Fits|🟡 Tight|🔴 Does not fit|⚪ Not applicable/);
  });

  it("renders 'Not applicable' for embedding workloads without crashing on NaN", () => {
    const scenario = buildScenario(b300, granite);
    const result = calculateScenario(scenario);
    const plan = createServingPlan(scenario, result);
    const md = buildScenarioReport({ hardware: b300, model: granite, scenario, result, plan });
    expect(md).toContain("⚪ Not applicable");
    // Step interval should render as an em dash, not "NaN"
    expect(md).not.toContain("NaN");
  });

  it("renders the warnings section when warnings exist", () => {
    const scenario = buildScenario(b300, qwen, { batchSize: 1, contextTokens: 1_000_000 });
    const result = calculateScenario(scenario);
    const plan = createServingPlan(scenario, result);
    const md = buildScenarioReport({ hardware: b300, model: qwen, scenario, result, plan });
    expect(md).toContain("## Warnings");
  });
});
