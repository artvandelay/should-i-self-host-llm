/**
 * Stream 0: inspect models.dev data quality before any filtering.
 * Read-only script — does not write anything.
 */

import { fetchModelsDev, extractOpenWeightModels, resolveParamsB } from "../src/modelsDev";

const MODELS_DEV_URL = "https://models.dev/api.json";

const SUSPECTED_NON_LLM_PATTERNS = [
  // embeddings & retrieval
  /\b(embed|embedding|retriev|reranker|rerank|nemoretriever)\b/i,
  // safety classifiers
  /\b(prompt[\s-]?guard|moderation|toxic)\b/i,
  /\bguard\b/i,
  // vision-only / OCR / VL
  /\b(paddle[\s-]?ocr|ocr|vl|cosmos[\s-]?transfer)\b/i,
  // speech / audio
  /\b(whisper|asr|tts|speech[\s-]?to[\s-]?text|text[\s-]?to[\s-]?speech)\b/i,
  // bio
  /\b(esm[12]|protein|biomed)\b/i,
  // code-completion-only
  /\b(codestral|devstral|prover[\s-]?v?\d)\b/i,
  // image / video gen
  /\b(flux|sdxl|stable[\s-]?diffusion|imagen|wan[\s-]?gen)\b/i,
];

async function main() {
  console.log("Fetching models.dev…\n");
  const payload = await fetchModelsDev();

  const totalProviders = Object.keys(payload ?? {}).length;
  let rawEntries = 0;
  for (const p of Object.values(payload ?? {})) {
    rawEntries += Object.keys(p.models ?? {}).length;
  }

  const openWeight = extractOpenWeightModels(payload ?? {});
  let regexMoe = 0;
  let regex = 0;
  let unknown = 0;
  const resolved = [];
  for (const m of openWeight) {
    const r = resolveParamsB(m.name, m.model_id);
    if (r.method === "regex_moe") regexMoe++;
    else if (r.method === "regex") regex++;
    else unknown++;
    resolved.push({ name: m.name, model_id: m.model_id, method: r.method, paramsB: r.paramsB });
  }

  console.log(`TOTAL: ${totalProviders} providers, ${rawEntries} raw entries, ${openWeight.length} open-weight unique by model_id\n`);
  console.log(`RESOLVED: ${regexMoe} regex_moe, ${regex} regex, ${unknown} unknown\n`);

  // Bottom 20 by params_b
  const withParams = resolved.filter((r) => r.paramsB != null).sort((a, b) => a.paramsB! - b.paramsB!);
  console.log("BOTTOM 20 BY params_b:");
  for (const m of withParams.slice(0, 20)) {
    console.log(`  ${String(m.paramsB!).padStart(6)}B  ${m.model_id}`);
  }
  console.log();

    console.log("TOP 10 BY params_b:");
  for (const m of withParams.slice(-10)) {
    console.log(`  ${String(m.paramsB!).padStart(6)}B  ${m.model_id}`);
  }
  console.log();

  // MoE entries
  const moeEntries = withParams.filter((m) => m.method === "regex_moe");
  console.log(`MoE ENTRIES (${moeEntries.length}):`);
  for (const m of moeEntries) {
    console.log(`  ${m.model_id}`);
  }
  console.log();

  // Suspected non-LLM counts
  console.log("SUSPECTED NON-LLM (regex match on name/model_id):");
  for (const re of SUSPECTED_NON_LLM_PATTERNS) {
    const hits = resolved.filter((m) => re.test(`${m.name} ${m.model_id}`));
    console.log(`  ${re.source.padEnd(35)} → ${hits.length} hits` + (hits.length ? `  e.g. "${hits[0].model_id}"` : ""));
  }
  console.log();
}

main().catch((e) => { console.error(e); process.exit(1); });
