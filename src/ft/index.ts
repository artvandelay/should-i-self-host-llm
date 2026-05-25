export type {
  FtMethod,
  Arch,
  ModelSpec,
  FtTraining,
  FtOptions,
  FtStage,
  FtWarning,
  FtCostBreakdown,
} from "./types";

export {
  FT_METHODS,
  FLOPS_PER_TOKEN_PER_PARAM,
  BASELINE_MFU,
  type FtMethodSpec,
} from "./methods";

export { ftVramGb, pickClusterOverhead, pickFtGpu } from "./hardware";

export { computeFtCost } from "./cost";

import type { FtMethod, ModelSpec, FtTraining, FtCostBreakdown } from "./types";
import { computeFtCost } from "./cost";

export interface FtInputs {
  num_examples: number;
  tokens_per_example: number;
  method: FtMethod;
  epochs: number;
  prep_cost_usd: number;
  /**
   * Experiments/failures multiplier — production FT campaigns require
   * multiple runs (HP sweeps, early-stopped failures, ablations).
   * 1.0 = single successful run (theoretical floor). 2-3 = realistic
   * range for production work. Default: 2.5.
   */
  experiments_multiplier: number;
  /**
   * Cluster overhead multiplier — accounts for inter-GPU and inter-node
   * communication cost (gradient sync, sharded backward). When omitted
   * (undefined), the engine auto-picks based on whether the FT workload
   * fits in a single H100 (1.0×), multi-GPU NVLink node (1.3×), or
   * needs multi-node (1.6×). Pass a number to override.
   */
  cluster_overhead?: number;
}

/**
 * Back-compat result shape used by the existing FtPanel + tests. Strips
 * `stages` and `warnings` from the rich `FtCostBreakdown`; new callers
 * should consume `computeFtCost` directly to get those.
 */
export type FtCapexResult = Omit<FtCostBreakdown, "stages" | "warnings">;

export function computeFtCapex(
  active_params_b: number,
  inputs: FtInputs,
  total_params_b?: number
): FtCapexResult {
  const spec: ModelSpec = {
    name: "",
    arch:
      total_params_b !== undefined && total_params_b !== active_params_b
        ? "moe"
        : "dense",
    total_params_b: total_params_b ?? active_params_b,
    active_params_b,
  };

  const training: FtTraining = {
    num_examples: inputs.num_examples,
    tokens_per_example: inputs.tokens_per_example,
    epochs: inputs.epochs,
    prep_cost_usd: inputs.prep_cost_usd,
    experiments_multiplier: inputs.experiments_multiplier,
  };

  const result = computeFtCost(spec, training, inputs.method, {
    cluster_overhead: inputs.cluster_overhead,
  });

  const { stages: _s, warnings: _w, ...rest } = result;
  return rest;
}
