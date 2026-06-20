Static physical specs. Prices are fetched live (see SKILL.md Phase 2).
Do not scrape this — it doesn't change.
Sources: NVIDIA / AMD datasheets, cross-referenced with
src/ft/ASSUMPTIONS.md §3 assumption 8 and 19.

## GPU spec reference

`mem_bandwidth_tbs` is HBM/global-memory bandwidth in TB/s. It is the binding
constraint for decode throughput (memory-bound), so the inference engine needs
it alongside `bf16_tflops` (compute-bound prefill / arithmetic-intensity).

| name | vram_gb | bf16_tflops | mem_bandwidth_tbs | gpus_per_node | notes |
| --- | --- | --- | --- | --- | --- |
| H100 SXM 80GB | 80  | 989  | 3.35 | 8 | NVIDIA Hopper, datasheet peak no sparsity |
| H200 SXM      | 141 | 989  | 4.8  | 8 | Same compute as H100, more+faster HBM3e |
| B200          | 192 | 2250 | 8.0  | 8 | NVIDIA Blackwell |
| A100 80GB     | 80  | 312  | 2.0  | 8 | NVIDIA Ampere (HBM2e) |
| A100 40GB     | 40  | 312  | 1.56 | 8 | NVIDIA Ampere (HBM2) |
| L40S          | 48  | 362  | 0.864| 8 | Ada inference-leaning (GDDR6) |
| L4            | 24  | 121  | 0.3  | 8 | Ada, low-power inference (GDDR6) |
| MI300X        | 192 | 1307 | 5.3  | 8 | AMD CDNA3 (HBM3) |
